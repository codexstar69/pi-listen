/**
 * Device detection — auto-detect hardware profile for smart model recommendations.
 *
 * Detects:
 * - RAM (container-aware via cgroup fallback)
 * - Raspberry Pi model (via /proc/device-tree/model)
 * - GPU (NVIDIA via nvidia-smi, Apple Metal via platform+arch)
 * - Container environment (Docker, cgroups)
 * - System locale for language auto-detection
 */

import * as os from "node:os";
import * as fs from "node:fs";
import { spawnSync } from "node:child_process";
import type { LocalModelInfo } from "./local";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface DeviceProfile {
	platform: NodeJS.Platform;
	arch: string;
	totalRamMB: number;
	freeRamMB: number;
	cpuCores: number;
	cpuModel: string;
	isRaspberryPi: boolean;
	piModel?: string;
	gpu: {
		hasNvidia: boolean;
		hasMetal: boolean;
		vramMB?: number;
		gpuName?: string;
	};
	isContainer: boolean;
	systemLocale: string;
}

export type ModelFitness = "recommended" | "compatible" | "warning" | "incompatible";

// ─── Detection ───────────────────────────────────────────────────────────────

/** Detect the current device profile. Synchronous — all checks are fast. */
export function detectDevice(): DeviceProfile {
	const platform = process.platform;
	const arch = process.arch;
	const cpuCores = os.cpus().length;
	const cpuModel = os.cpus()[0]?.model || "unknown";

	// RAM — container-aware
	const isContainer = detectContainer();
	const hostRamMB = Math.round(os.totalmem() / (1024 * 1024));
	const totalRamMB = isContainer ? getContainerRamMB(hostRamMB) : hostRamMB;
	const freeRamMB = Math.round(os.freemem() / (1024 * 1024));

	// Raspberry Pi
	const piInfo = detectRaspberryPi();

	// GPU
	const gpu = detectGPU(platform, arch);

	// Locale
	const systemLocale = detectLocale();

	return {
		platform,
		arch,
		totalRamMB,
		freeRamMB,
		cpuCores,
		cpuModel,
		isRaspberryPi: piInfo.isRPi,
		piModel: piInfo.model,
		gpu,
		isContainer,
		systemLocale,
	};
}

// ─── Model fitness scoring ───────────────────────────────────────────────────

/**
 * Score how well a model fits this device.
 * Uses runtime RAM estimates (~2.5x model file size for ONNX inference).
 */
export function getModelFitness(model: LocalModelInfo, device: DeviceProfile): ModelFitness {
	const runtimeRamMB = model.runtimeRamMB ?? estimateRuntimeRam(model.sizeBytes);
	const ratio = runtimeRamMB / device.totalRamMB;

	if (ratio > 0.8) return "incompatible";
	if (runtimeRamMB > device.freeRamMB) return "warning";
	if (ratio > 0.5) return "compatible";
	return "recommended";
}

/** Estimate runtime RAM from download size (bytes) — ~2.5x model file size. */
function estimateRuntimeRam(sizeBytes?: number): number {
	if (!sizeBytes) return 500; // Conservative default
	return Math.round((sizeBytes / (1024 * 1024)) * 2.5);
}

/**
 * Auto-recommend the best model for a device + language combination.
 * Prioritizes: language fit → device fitness → accuracy (larger is better within recommended).
 */
export function autoRecommendModel(
	models: LocalModelInfo[],
	device: DeviceProfile,
	language: string,
): LocalModelInfo | undefined {
	// Filter by language support
	const langModels = models.filter(m => modelSupportsLanguage(m, language));
	if (langModels.length === 0) return undefined;

	// Score each model
	const scored = langModels.map(m => ({
		model: m,
		fitness: getModelFitness(m, device),
		size: m.sizeBytes || 0,
	}));

	// Prefer recommended > compatible > warning, then largest within tier (more accurate)
	const fitnessOrder: Record<ModelFitness, number> = {
		recommended: 0,
		compatible: 1,
		warning: 2,
		incompatible: 3,
	};

	scored.sort((a, b) => {
		const fitDiff = fitnessOrder[a.fitness] - fitnessOrder[b.fitness];
		if (fitDiff !== 0) return fitDiff;
		// Within same fitness tier, prefer larger (more accurate)
		return b.size - a.size;
	});

	// Don't recommend incompatible models
	const best = scored[0];
	if (best && best.fitness !== "incompatible") return best.model;

	// Fallback: smallest model regardless
	return scored[scored.length - 1]?.model;
}

/** Check if a model supports a given language code. */
function modelSupportsLanguage(model: LocalModelInfo, langCode: string): boolean {
	const base = langCode.split("-")[0];
	switch (model.langSupport) {
		case "whisper":
		case "parakeet-multi":
			return true; // Multilingual
		case "english-only":
			return base === "en";
		case "russian-only":
			return base === "ru";
		case "sensevoice":
			return ["zh", "en", "ja", "ko", "yue"].includes(base!);
		case "single-ar": return base === "ar";
		case "single-zh": return base === "zh";
		case "single-ja": return base === "ja";
		case "single-ko": return base === "ko";
		case "single-uk": return base === "uk";
		case "single-vi": return base === "vi";
		case "single-es": return base === "es";
		default:
			return true;
	}
}

/** Format device profile as a short summary string. */
export function formatDeviceSummary(device: DeviceProfile): string {
	const parts: string[] = [];

	// RAM
	const ramGB = (device.totalRamMB / 1024).toFixed(1);
	parts.push(`${ramGB} GB RAM`);

	// Platform/arch
	parts.push(device.arch);

	// RPi
	if (device.isRaspberryPi && device.piModel) {
		parts.push(device.piModel);
	} else {
		const platformNames: Record<string, string> = {
			darwin: "macOS",
			linux: "Linux",
			win32: "Windows",
		};
		parts.push(platformNames[device.platform] || device.platform);
	}

	// GPU
	if (device.gpu.hasNvidia && device.gpu.gpuName) {
		parts.push(device.gpu.gpuName);
	} else if (device.gpu.hasMetal) {
		parts.push("Apple Silicon");
	}

	// Container
	if (device.isContainer) {
		parts.push("container");
	}

	return parts.join(", ");
}

// ─── Internal detection helpers ──────────────────────────────────────────────

function detectContainer(): boolean {
	try {
		if (fs.existsSync("/.dockerenv")) return true;
		const cgroup = fs.readFileSync("/proc/1/cgroup", "utf-8");
		if (cgroup.includes("docker") || cgroup.includes("kubepods") || cgroup.includes("containerd")) return true;
	} catch {
		// Not Linux or no permissions
	}
	return false;
}

function getContainerRamMB(hostRamMB: number): number {
	// Try cgroup v2 first, then v1
	const paths = [
		"/sys/fs/cgroup/memory.max",           // cgroup v2
		"/sys/fs/cgroup/memory/memory.limit_in_bytes", // cgroup v1
	];
	for (const p of paths) {
		try {
			const raw = fs.readFileSync(p, "utf-8").trim();
			if (raw === "max" || raw === "9223372036854775807") continue; // Unlimited
			const bytes = parseInt(raw, 10);
			if (Number.isFinite(bytes) && bytes > 0) {
				return Math.min(Math.round(bytes / (1024 * 1024)), hostRamMB);
			}
		} catch {
			// File not accessible
		}
	}
	return hostRamMB;
}

function detectRaspberryPi(): { isRPi: boolean; model?: string } {
	// Method 1: /proc/device-tree/model (most reliable)
	try {
		const model = fs.readFileSync("/proc/device-tree/model", "utf-8").replace(/\0/g, "").trim();
		if (model.toLowerCase().includes("raspberry pi")) {
			return { isRPi: true, model };
		}
	} catch {
		// Not available
	}

	// Method 2: /proc/cpuinfo BCM chip
	try {
		const cpuinfo = fs.readFileSync("/proc/cpuinfo", "utf-8");
		if (cpuinfo.includes("BCM2")) {
			const hwMatch = cpuinfo.match(/^Hardware\s*:\s*(.+)$/m);
			return { isRPi: true, model: hwMatch?.[1]?.trim() };
		}
	} catch {
		// Not available
	}

	// Method 3: ARM64 + Debian/Raspbian heuristic
	if (process.arch === "arm64" || process.arch === "arm") {
		try {
			const release = fs.readFileSync("/etc/os-release", "utf-8");
			if (release.includes("Raspbian") || release.includes("raspberry")) {
				return { isRPi: true };
			}
		} catch {
			// Not available
		}
	}

	return { isRPi: false };
}

function detectGPU(platform: NodeJS.Platform, arch: string): DeviceProfile["gpu"] {
	const result: DeviceProfile["gpu"] = {
		hasNvidia: false,
		hasMetal: false,
	};

	// Apple Metal — macOS + ARM64
	if (platform === "darwin" && arch === "arm64") {
		result.hasMetal = true;
	}

	// NVIDIA — try nvidia-smi (5s timeout)
	try {
		const nv = spawnSync("nvidia-smi", [
			"--query-gpu=name,memory.total",
			"--format=csv,noheader,nounits",
		], { timeout: 5000, encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] });

		if (nv.status === 0 && nv.stdout) {
			const line = nv.stdout.trim().split("\n")[0];
			if (line) {
				const [name, vram] = line.split(",").map(s => s?.trim());
				result.hasNvidia = true;
				result.gpuName = name;
				result.vramMB = vram ? parseInt(vram, 10) : undefined;
			}
		}
	} catch {
		// nvidia-smi not available
	}

	return result;
}

function detectLocale(): string {
	try {
		const resolved = Intl.DateTimeFormat().resolvedOptions().locale;
		if (resolved) return resolved;
	} catch {
		// Fallback
	}

	// Try environment
	const envLocale = process.env.LANG || process.env.LC_ALL || process.env.LC_MESSAGES;
	if (envLocale) {
		// "en_US.UTF-8" → "en-US"
		const base = envLocale.split(".")[0];
		return base?.replace("_", "-") || "en";
	}

	return "en";
}

/** Extract the base language code from a system locale (e.g. "en-US" → "en"). */
export function localeToLanguageCode(locale: string): string {
	return locale.split("-")[0] || "en";
}
