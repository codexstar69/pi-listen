/**
 * Model download manager — auto-download ONNX models for local transcription.
 *
 * Downloads from sherpa-onnx GitHub releases and HuggingFace.
 * Defaults to int8 quantized models for smaller downloads and lower RAM usage.
 *
 * Storage: ~/.pi/models/{modelId}/
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ModelDownloadConfig {
	/** Model identifier (e.g., "whisper-small") */
	modelId: string;
	/** Map of role → download URL */
	files: Record<string, string>;
	/** Expected total download size in bytes (for progress reporting) */
	totalSizeBytes: number;
}

export interface DownloadProgress {
	downloadedBytes: number;
	totalBytes: number;
	file: string;
	fileIndex: number;
	totalFiles: number;
}

// ─── Paths ───────────────────────────────────────────────────────────────────

/** Get the base models directory (~/.pi/models/) */
export function getModelsDir(): string {
	return path.join(os.homedir(), ".pi", "models");
}

/** Get the directory for a specific model */
export function getModelDir(modelId: string): string {
	return path.join(getModelsDir(), modelId);
}

/** Get the path for a specific model file */
export function getModelPath(modelId: string): string | null {
	const dir = getModelDir(modelId);
	if (!fs.existsSync(dir)) return null;
	return dir;
}

// ─── Status checks ───────────────────────────────────────────────────────────

/** Check if a model is fully downloaded (all expected files present). */
export function isModelDownloaded(modelId: string, expectedFiles: Record<string, string>): boolean {
	const dir = getModelDir(modelId);
	if (!fs.existsSync(dir)) return false;

	for (const role of Object.keys(expectedFiles)) {
		const filename = fileNameFromUrl(expectedFiles[role]!);
		const filePath = path.join(dir, filename);
		if (!fs.existsSync(filePath)) return false;
	}
	return true;
}

/** List downloaded models with disk usage. */
export function getDownloadedModels(): { id: string; sizeMB: number }[] {
	const baseDir = getModelsDir();
	if (!fs.existsSync(baseDir)) return [];

	const results: { id: string; sizeMB: number }[] = [];
	try {
		const entries = fs.readdirSync(baseDir, { withFileTypes: true });
		for (const entry of entries) {
			if (!entry.isDirectory()) continue;
			const modelDir = path.join(baseDir, entry.name);
			const size = getDirSizeMB(modelDir);
			results.push({ id: entry.name, sizeMB: size });
		}
	} catch {
		// Permission error
	}
	return results;
}

/** Delete a downloaded model. */
export function deleteModel(modelId: string): boolean {
	const dir = getModelDir(modelId);
	if (!fs.existsSync(dir)) return false;
	try {
		fs.rmSync(dir, { recursive: true, force: true });
		return true;
	} catch {
		return false;
	}
}

// ─── Download ────────────────────────────────────────────────────────────────

/**
 * Download all model files for a given model config.
 * Returns the model directory path.
 *
 * Features:
 * - Progress callbacks per file and overall
 * - Resume support via HTTP Range headers (partial downloads)
 * - Atomic writes (download to .tmp, rename on success)
 * - Abort support via AbortSignal
 */
export async function downloadModel(
	config: ModelDownloadConfig,
	onProgress?: (progress: DownloadProgress) => void,
	signal?: AbortSignal,
): Promise<string> {
	const dir = getModelDir(config.modelId);
	fs.mkdirSync(dir, { recursive: true });

	const roles = Object.keys(config.files);
	let overallDownloaded = 0;

	for (let i = 0; i < roles.length; i++) {
		const role = roles[i]!;
		const url = config.files[role]!;
		const filename = fileNameFromUrl(url);
		const filePath = path.join(dir, filename);
		const tmpPath = filePath + ".tmp";

		// Skip if already downloaded
		if (fs.existsSync(filePath)) {
			const stat = fs.statSync(filePath);
			overallDownloaded += stat.size;
			continue;
		}

		// Check for partial download (resume support)
		let startByte = 0;
		if (fs.existsSync(tmpPath)) {
			startByte = fs.statSync(tmpPath).size;
			overallDownloaded += startByte;
		}

		const headers: Record<string, string> = {};
		if (startByte > 0) {
			headers["Range"] = `bytes=${startByte}-`;
		}

		const resp = await fetch(url, {
			headers,
			signal,
			redirect: "follow",
		});

		if (!resp.ok && resp.status !== 206) {
			throw new Error(`Download failed: HTTP ${resp.status} for ${filename}`);
		}

		const contentLength = parseInt(resp.headers.get("content-length") || "0", 10);
		const totalFileSize = startByte + contentLength;

		if (!resp.body) throw new Error(`No response body for ${filename}`);

		const writeStream = fs.createWriteStream(tmpPath, { flags: startByte > 0 ? "a" : "w" });
		const reader = resp.body.getReader();
		let fileDownloaded = startByte;

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				writeStream.write(Buffer.from(value));
				fileDownloaded += value.byteLength;
				overallDownloaded += value.byteLength;

				onProgress?.({
					downloadedBytes: overallDownloaded,
					totalBytes: config.totalSizeBytes,
					file: filename,
					fileIndex: i,
					totalFiles: roles.length,
				});
			}
		} finally {
			writeStream.end();
			await new Promise<void>((resolve, reject) => {
				writeStream.on("finish", resolve);
				writeStream.on("error", reject);
			});
		}

		// Atomic rename .tmp → final
		fs.renameSync(tmpPath, filePath);
	}

	return dir;
}

/**
 * Ensure a model is downloaded, downloading if needed.
 * This is the main entry point for the transcription engine.
 */
export async function ensureModelDownloaded(
	modelId: string,
	expectedFiles: Record<string, string>,
	totalSizeBytes: number,
	onProgress?: (progress: DownloadProgress) => void,
	signal?: AbortSignal,
): Promise<string> {
	if (isModelDownloaded(modelId, expectedFiles)) {
		return getModelDir(modelId);
	}

	return downloadModel(
		{ modelId, files: expectedFiles, totalSizeBytes },
		onProgress,
		signal,
	);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Extract filename from a URL. */
function fileNameFromUrl(url: string): string {
	const urlPath = new URL(url).pathname;
	return path.basename(urlPath);
}

/** Get total size of a directory in bytes. */
function getDirSizeBytes(dirPath: string): number {
	let total = 0;
	try {
		const entries = fs.readdirSync(dirPath, { withFileTypes: true });
		for (const entry of entries) {
			const fullPath = path.join(dirPath, entry.name);
			if (entry.isFile()) {
				total += fs.statSync(fullPath).size;
			} else if (entry.isDirectory()) {
				total += getDirSizeBytes(fullPath);
			}
		}
	} catch {
		// Permission error
	}
	return total;
}

/** Get total size of a directory in MB. */
function getDirSizeMB(dirPath: string): number {
	return Math.round(getDirSizeBytes(dirPath) / (1024 * 1024));
}
