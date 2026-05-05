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
	// os.cpus() can return undefined on Alpine/restricted containers
	const cpuList = os.cpus() || [];
	const cpuCores = cpuList.length || 1;
	const cpuModel = cpuList[0]?.model || "unknown";

	// RAM — container-aware
	const isContainer = detectContainer();
	const hostRamMB = Math.round(os.totalmem() / (1024 * 1024));
	const totalRamMB = isContainer ? getContainerRamMB(hostRamMB) : hostRamMB;
	// Use MemAvailable from /proc/meminfo on Linux (more accurate than os.freemem()
	// which returns MemFree, ignoring reclaimable buffer/cache memory)
	const freeRamMB = getAvailableRamMB();

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
 *
 * "recommended" is reserved for preferred models (best-in-class for their
 * language/use case) that fit comfortably. All other runnable models are
 * "compatible". This prevents every model from showing [recommended] on
 * machines with plenty of RAM.
 */
export function getModelFitness(model: LocalModelInfo, device: DeviceProfile): ModelFitness {
	const runtimeRamMB = model.runtimeRamMB ?? estimateRuntimeRam(model.sizeBytes);
	const ratio = runtimeRamMB / device.totalRamMB;

	// >80% of total RAM — won't run
	if (ratio > 0.8) return "incompatible";
	// >60% of total RAM — will run but may cause swapping
	if (ratio > 0.6) return "warning";
	// Preferred models that fit comfortably get "recommended"
	if (model.preferred && ratio < 0.4) return "recommended";
	return "compatible";
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

/**
 * Get available RAM in MB — uses /proc/meminfo MemAvailable on Linux
 * (includes reclaimable buffer/cache), falls back to os.freemem() elsewhere.
 *
 * os.freemem() returns MemFree on Linux, which excludes buffer/cache and is
 * often ~500MB even on a 32GB machine. MemAvailable is what the kernel considers
 * actually available for new processes.
 */
function getAvailableRamMB(): number {
	if (process.platform === "linux") {
		try {
			const meminfo = fs.readFileSync("/proc/meminfo", "utf-8");
			const match = meminfo.match(/^MemAvailable:\s+(\d+)\s+kB$/m);
			if (match) {
				return Math.round(parseInt(match[1]!, 10) / 1024);
			}
		} catch {
			// Fallback to os.freemem()
		}
	}
	return Math.round(os.freemem() / (1024 * 1024));
}

function detectContainer(): boolean {
	try {
		if (fs.existsSync("/.dockerenv")) return true;
		if (fs.existsSync("/run/.containerenv")) return true; // Podman
		const cgroup = fs.readFileSync("/proc/1/cgroup", "utf-8");
		if (cgroup.includes("docker") || cgroup.includes("kubepods") || cgroup.includes("containerd")) return true;
		// cgroup v2: check /proc/self/mountinfo for container indicators
		if (cgroup.trim() === "0::/") {
			try {
				const mountinfo = fs.readFileSync("/proc/self/mountinfo", "utf-8");
				if (mountinfo.includes("/docker/") || mountinfo.includes("/containers/")) return true;
			} catch {
				// Not accessible
			}
		}
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
			// "max" = cgroup v2 unlimited
		if (raw === "max") continue;
			// cgroup v1 unlimited: LLONG_MAX or page-aligned variants (64-bit and 32-bit)
			// Use string comparison to avoid parseInt precision loss on values > MAX_SAFE_INTEGER
			if (raw === "9223372036854775807" || raw === "9223372036854771712") continue;
			const bytes = parseInt(raw, 10);
			if (!Number.isFinite(bytes) || bytes <= 0) continue;
			// Heuristic: if cgroup value exceeds host RAM, treat as unlimited
			// Catches 32-bit sentinels (~2 GB) and any non-standard page-aligned variants
			const mb = Math.round(bytes / (1024 * 1024));
			if (mb >= hostRamMB) continue;
			return mb;
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
			// Prefer "Model" field (human-readable) over "Hardware" (just shows BCM2835 for all models)
			const modelMatch = cpuinfo.match(/^Model\s*:\s*(.+)$/m);
			const hwMatch = cpuinfo.match(/^Hardware\s*:\s*(.+)$/m);
			return { isRPi: true, model: modelMatch?.[1]?.trim() || hwMatch?.[1]?.trim() };
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

	// NVIDIA — skip expensive nvidia-smi if no GPU device file exists (Linux)
	if (platform === "linux" && !fs.existsSync("/dev/nvidiactl")) {
		return result;
	}

	// NVIDIA — try nvidia-smi (2s timeout)
	try {
		const nv = spawnSync("nvidia-smi", [
			"--query-gpu=name,memory.total",
			"--format=csv,noheader,nounits",
		], { timeout: 2000, encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] });

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

// ─── CUDA detection (cheap, env-authoritative) ───────────────────────────────

/**
 * Tokens that disable CUDA when set as the env-var value. Match is
 * case-insensitive after `.trim()`:
 *   - ""              → disabled (explicit empty / whitespace-only)
 *   - "-1"            → disabled (CUDA convention)
 *   - "none" / "void" → disabled (NVIDIA conventions)
 *
 * Anything else (e.g. "0", "all", "0,1,2") enables CUDA. To re-enable
 * CUDA after disabling via env, unset the variable entirely or set it
 * to a device id like "0".
 */
const CUDA_DISABLE_VALUES = new Set(["", "-1", "none", "void"]);

/**
 * Cheap CUDA-availability check used by `engine-factory` to decide whether
 * a CUDA-only model is selectable on the current runtime.
 *
 * Env-var semantics (from round-2/3 godspeed review):
 *   - When CUDA_VISIBLE_DEVICES or NVIDIA_VISIBLE_DEVICES is SET, its value
 *     is authoritative — we do NOT fall through to native probing. This
 *     lets users disable CUDA explicitly (`CUDA_VISIBLE_DEVICES=""` or
 *     `="-1"`) without `/dev/nvidia0` overriding their intent.
 *   - When neither var is set, we probe native paths. Linux only — Windows
 *     has no shipped whisper.cpp CUDA prebuilt today; revisit when it does.
 *
 * Distinct from `detectGPU()` above which spawns `nvidia-smi` to fetch
 * VRAM/name. This function never spawns; it's safe to call repeatedly.
 */
export function hasCUDA(
	env: NodeJS.ProcessEnv = process.env,
	platform: NodeJS.Platform = process.platform,
): boolean {
	// Step 1: honor explicit env override on every platform.
	const visible = env.CUDA_VISIBLE_DEVICES ?? env.NVIDIA_VISIBLE_DEVICES;
	if (visible !== undefined) {
		return !CUDA_DISABLE_VALUES.has(visible.trim().toLowerCase());
	}
	// Step 2: env var unset — probe native paths on platforms we ship for.
	if (platform === "linux") {
		return fs.existsSync("/dev/nvidia0") ||
			fs.existsSync("/proc/driver/nvidia/version");
	}
	// Windows is intentionally NOT probed — see Non-goals (no verified
	// win32-x64-CUDA prebuilt). Add detection only when prebuilts ship.
	return false;
}

/**
 * Whether the current `smart-whisper` binding was built with CUDA support.
 *
 * `hasCUDA()` only knows whether the SYSTEM has a CUDA-capable GPU and
 * driver. It cannot detect whether the user's `node-gyp rebuild` of
 * `smart-whisper` actually compiled the CUDA backend in (which requires
 * `WHISPER_CUDA=1` at install time). A user with CUDA hardware but a
 * default install would still get CPU-only inference, just slowly.
 *
 * Round-2 audit fix: gate `requiresCUDA: true` model selection on BOTH
 * `hasCUDA()` AND `cudaBuildCapable()`. The latter is signaled via the
 * `WHISPER_CUDA` env var — users who installed with
 * `WHISPER_CUDA=1 bun install` typically keep the var exported in their
 * shell config or systemd unit, so it persists into the runtime
 * environment. If they don't keep it set, CUDA models hide from the
 * picker and they default to sherpa-onnx until they re-export it.
 *
 * This isn't a true binding-introspection probe (smart-whisper doesn't
 * expose one); it's a user-asserted capability flag that matches the
 * documented install workflow. A future smart-whisper version could
 * expose `getBackendCapabilities()` and we'd switch to that.
 */
export function cudaBuildCapable(env: NodeJS.ProcessEnv = process.env): boolean {
	return env.WHISPER_CUDA === "1";
}
