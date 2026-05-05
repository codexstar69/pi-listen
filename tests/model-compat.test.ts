import { describe, expect, test } from "bun:test";
import {
	currentVoiceRuntimeEnv,
	describeUnsupported,
	isModelSupported,
	type VoiceRuntimeEnv,
} from "../extensions/voice/model-compat";
import type { LocalModelInfo } from "../extensions/voice/local";

// Stub catalog entries for the support matrix. Only the fields the
// predicate inspects need to be set; everything else is filler.
function model(
	id: string,
	supports?: LocalModelInfo["supports"],
	engine?: LocalModelInfo["engine"],
): LocalModelInfo {
	return {
		id,
		name: id,
		size: "0",
		sizeBytes: 0,
		runtimeRamMB: 0,
		notes: "",
		langSupport: "english-only",
		tier: "edge",
		accuracy: 1,
		speed: 1,
		sherpaModel: { type: "moonshine", files: {}, downloadUrls: {} },
		...(supports !== undefined ? { supports } : {}),
		...(engine !== undefined ? { engine } : {}),
	};
}

// `cudaBuildCapable` is the new round-2 audit gate: smart-whisper was
// built with CUDA support. CUDA-required models need BOTH hasCUDA AND
// cudaBuildCapable to be selectable.
const mac: VoiceRuntimeEnv = { platform: "darwin", arch: "arm64", hasCUDA: false, cudaBuildCapable: false };
const intelMac: VoiceRuntimeEnv = { platform: "darwin", arch: "x64", hasCUDA: false, cudaBuildCapable: false };
const linuxCpu: VoiceRuntimeEnv = { platform: "linux", arch: "x64", hasCUDA: false, cudaBuildCapable: false };
// Full CUDA: hardware AND build-capable.
const linuxCuda: VoiceRuntimeEnv = { platform: "linux", arch: "x64", hasCUDA: true, cudaBuildCapable: true };
const linuxArmCuda: VoiceRuntimeEnv = { platform: "linux", arch: "arm64", hasCUDA: true, cudaBuildCapable: true };

describe("isModelSupported", () => {
	test("undefined supports → universally supported (back-compat default)", () => {
		const m = model("legacy");
		expect(isModelSupported(m, mac)).toBe(true);
		expect(isModelSupported(m, linuxCpu)).toBe(true);
		expect(isModelSupported(m, linuxArmCuda)).toBe(true);
	});

	test("empty supports array → universally supported", () => {
		const m = model("legacy", []);
		expect(isModelSupported(m, mac)).toBe(true);
	});

	test("platform-only constraint matches platform across archs", () => {
		const m = model("any-darwin", [{ platform: "darwin" }]);
		expect(isModelSupported(m, mac)).toBe(true);
		expect(isModelSupported(m, intelMac)).toBe(true);
		expect(isModelSupported(m, linuxCpu)).toBe(false);
	});

	test("platform + arch constraint requires both", () => {
		const m = model("metal", [{ platform: "darwin", arch: "arm64" }]);
		expect(isModelSupported(m, mac)).toBe(true);
		expect(isModelSupported(m, intelMac)).toBe(false);
		expect(isModelSupported(m, linuxCpu)).toBe(false);
	});

	test("requiresCUDA gates on BOTH hasCUDA AND cudaBuildCapable", () => {
		const m = model("cuda-x64", [{ platform: "linux", arch: "x64", requiresCUDA: true }]);
		// Full CUDA — both flags true → supported
		expect(isModelSupported(m, linuxCuda)).toBe(true);
		// No hardware → not supported
		expect(isModelSupported(m, linuxCpu)).toBe(false);
		// Hardware present but smart-whisper wasn't built with CUDA →
		// catalog hides the model so the user doesn't pick a CUDA-only
		// entry that would silently run on CPU.
		const cudaHardwareNoBuild: VoiceRuntimeEnv = {
			platform: "linux", arch: "x64", hasCUDA: true, cudaBuildCapable: false,
		};
		expect(isModelSupported(m, cudaHardwareNoBuild)).toBe(false);
	});

	test("arch and CUDA checks are independent (no short-circuit bypass)", () => {
		// Round-9 grok concern: arch validation must NOT be bypassed when
		// CUDA happens to be available.
		const m = model("cuda-x64-only", [{ platform: "linux", arch: "x64", requiresCUDA: true }]);
		expect(isModelSupported(m, linuxArmCuda)).toBe(false);
	});

	test("multiple support entries — any match wins", () => {
		const m = model("metal-or-cuda", [
			{ platform: "darwin", arch: "arm64" },
			{ platform: "linux", arch: "x64", requiresCUDA: true },
		]);
		expect(isModelSupported(m, mac)).toBe(true);
		expect(isModelSupported(m, linuxCuda)).toBe(true);
		expect(isModelSupported(m, linuxCpu)).toBe(false);
		expect(isModelSupported(m, intelMac)).toBe(false);
	});
});

describe("describeUnsupported", () => {
	test("includes platform-arch pair", () => {
		const m = model("metal", [{ platform: "darwin", arch: "arm64" }]);
		expect(describeUnsupported(m, linuxCpu)).toContain("linux-x64");
		expect(describeUnsupported(m, linuxCpu)).toContain("metal");
	});

	test("notes CUDA hardware unavailability when applicable", () => {
		const m = model("cuda", [{ platform: "linux", arch: "x64", requiresCUDA: true }]);
		expect(describeUnsupported(m, linuxCpu)).toContain("CUDA hardware unavailable");
		expect(describeUnsupported(m, linuxCuda)).not.toContain("unavailable");
	});

	test("notes CUDA-build-capable mismatch when hardware present but binding wasn't", () => {
		const m = model("cuda", [{ platform: "linux", arch: "x64", requiresCUDA: true }]);
		const cudaHardwareNoBuild: VoiceRuntimeEnv = {
			platform: "linux", arch: "x64", hasCUDA: true, cudaBuildCapable: false,
		};
		expect(describeUnsupported(m, cudaHardwareNoBuild))
			.toContain("WHISPER_CUDA=1");
	});
});

describe("currentVoiceRuntimeEnv", () => {
	test("returns process.platform / process.arch", () => {
		const env = currentVoiceRuntimeEnv();
		expect(env.platform).toBe(process.platform);
		expect(env.arch).toBe(process.arch);
		expect(typeof env.hasCUDA).toBe("boolean");
	});
});
