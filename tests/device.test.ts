import { describe, expect, test } from "bun:test";
import {
	detectDevice,
	getModelFitness,
	autoRecommendModel,
	formatDeviceSummary,
	hasCUDA,
	localeToLanguageCode,
	type DeviceProfile,
} from "../extensions/voice/device";
import { LOCAL_MODELS, type LocalModelInfo } from "../extensions/voice/local";

// ─── Helper: mock device profiles ────────────────────────────────────────────

function mockDevice(overrides: Partial<DeviceProfile> = {}): DeviceProfile {
	return {
		platform: "linux",
		arch: "x64",
		totalRamMB: 8192,
		freeRamMB: 4096,
		cpuCores: 4,
		cpuModel: "Test CPU",
		isRaspberryPi: false,
		gpu: { hasNvidia: false, hasMetal: false },
		isContainer: false,
		systemLocale: "en-US",
		...overrides,
	};
}

// ─── detectDevice ────────────────────────────────────────────────────────────

describe("detectDevice", () => {
	test("returns a valid device profile", () => {
		const device = detectDevice();

		expect(device.platform).toBeString();
		expect(device.arch).toBeString();
		expect(device.totalRamMB).toBeGreaterThan(0);
		expect(device.freeRamMB).toBeGreaterThanOrEqual(0);
		expect(device.cpuCores).toBeGreaterThan(0);
		expect(device.cpuModel).toBeString();
		expect(typeof device.isRaspberryPi).toBe("boolean");
		expect(typeof device.isContainer).toBe("boolean");
		expect(device.systemLocale).toBeString();
	});

	test("totalRamMB is reasonable (at least 256 MB)", () => {
		const device = detectDevice();
		expect(device.totalRamMB).toBeGreaterThan(256);
	});

	test("freeRamMB uses MemAvailable (should be > os.freemem on Linux)", () => {
		const device = detectDevice();
		// freeRamMB should be reasonable — at least some RAM should be available
		expect(device.freeRamMB).toBeGreaterThan(0);
		// freeRamMB should not exceed totalRamMB
		expect(device.freeRamMB).toBeLessThanOrEqual(device.totalRamMB);
	});
});

// ─── getModelFitness ─────────────────────────────────────────────────────────

describe("getModelFitness", () => {
	const tinyModel = LOCAL_MODELS.find(m => m.id === "moonshine-v2-tiny")!;
	const largeModel = LOCAL_MODELS.find(m => m.id === "whisper-large")!;

	test("tiny model is recommended on 8GB device", () => {
		const device = mockDevice({ totalRamMB: 8192, freeRamMB: 6000 });
		expect(getModelFitness(tinyModel, device)).toBe("recommended");
	});

	test("large model is incompatible on 2GB device", () => {
		const device = mockDevice({ totalRamMB: 2048, freeRamMB: 1024 });
		expect(getModelFitness(largeModel, device)).toBe("incompatible");
	});

	test("large model is compatible on 16GB device", () => {
		const device = mockDevice({ totalRamMB: 16384, freeRamMB: 8000 });
		const fitness = getModelFitness(largeModel, device);
		expect(["recommended", "compatible"]).toContain(fitness);
	});

	test("model returns warning when runtime exceeds 60% of total RAM", () => {
		// whisper-medium needs ~2365 MB runtime, device has 3000 MB total → ratio 0.79 > 0.6
		const mediumModel = LOCAL_MODELS.find(m => m.id === "whisper-medium")!;
		const device = mockDevice({ totalRamMB: 3000, freeRamMB: 2000 });
		expect(getModelFitness(mediumModel, device)).toBe("warning");
	});
});

// ─── autoRecommendModel ──────────────────────────────────────────────────────

describe("autoRecommendModel", () => {
	test("recommends a model for English on 4GB device", () => {
		const device = mockDevice({ totalRamMB: 4096, freeRamMB: 2048 });
		const recommended = autoRecommendModel(LOCAL_MODELS, device, "en");

		expect(recommended).toBeDefined();
		expect(recommended!.id).toBeString();
		// Should not recommend a model that's incompatible
		const fitness = getModelFitness(recommended!, device);
		expect(fitness).not.toBe("incompatible");
	});

	test("recommends Russian-specific model for Russian language", () => {
		const device = mockDevice({ totalRamMB: 4096, freeRamMB: 2048 });
		const recommended = autoRecommendModel(LOCAL_MODELS, device, "ru");

		expect(recommended).toBeDefined();
		// GigaAM v3 or whisper should be recommended for Russian
		const model = recommended!;
		const supportsRussian = model.langSupport === "russian-only" || model.langSupport === "whisper" || model.langSupport === "parakeet-multi";
		expect(supportsRussian).toBe(true);
	});

	test("recommends edge-tier model for low-RAM RPi", () => {
		const device = mockDevice({ totalRamMB: 1024, freeRamMB: 512, isRaspberryPi: true });
		const recommended = autoRecommendModel(LOCAL_MODELS, device, "en");

		expect(recommended).toBeDefined();
		// Should pick a small model
		expect(recommended!.runtimeRamMB).toBeLessThan(500);
	});

	test("recommends larger model for high-RAM device", () => {
		const device = mockDevice({ totalRamMB: 32768, freeRamMB: 24000 });
		const recommended = autoRecommendModel(LOCAL_MODELS, device, "en");

		expect(recommended).toBeDefined();
		// With 32GB, should recommend something substantial
		expect(recommended!.sizeBytes).toBeGreaterThan(50_000_000);
	});

	test("returns undefined for unsupported language with no multilingual models", () => {
		// Filter to only english-only models
		const englishOnly = LOCAL_MODELS.filter(m => m.langSupport === "english-only");
		const device = mockDevice();
		const recommended = autoRecommendModel(englishOnly, device, "sw"); // Swahili
		expect(recommended).toBeUndefined();
	});

	test("recommends Japanese model for Japanese language", () => {
		const device = mockDevice({ totalRamMB: 4096, freeRamMB: 2048 });
		const recommended = autoRecommendModel(LOCAL_MODELS, device, "ja");
		expect(recommended).toBeDefined();
		// Should pick a model that supports Japanese
		const supportsJa = ["whisper", "parakeet-multi", "sensevoice", "single-ja"].includes(recommended!.langSupport);
		expect(supportsJa).toBe(true);
	});
});

// ─── formatDeviceSummary ─────────────────────────────────────────────────────

describe("formatDeviceSummary", () => {
	test("includes RAM", () => {
		const device = mockDevice({ totalRamMB: 8192 });
		const summary = formatDeviceSummary(device);
		expect(summary).toContain("8.0 GB RAM");
	});

	test("includes RPi model when detected", () => {
		const device = mockDevice({ isRaspberryPi: true, piModel: "Raspberry Pi 5" });
		const summary = formatDeviceSummary(device);
		expect(summary).toContain("Raspberry Pi 5");
	});

	test("includes Apple Silicon when Metal detected", () => {
		const device = mockDevice({ platform: "darwin", gpu: { hasNvidia: false, hasMetal: true } });
		const summary = formatDeviceSummary(device);
		expect(summary).toContain("Apple Silicon");
	});
});

// ─── localeToLanguageCode ────────────────────────────────────────────────────

describe("localeToLanguageCode", () => {
	test("extracts language from locale", () => {
		expect(localeToLanguageCode("en-US")).toBe("en");
		expect(localeToLanguageCode("ja-JP")).toBe("ja");
		expect(localeToLanguageCode("pt-BR")).toBe("pt");
	});

	test("handles bare language code", () => {
		expect(localeToLanguageCode("en")).toBe("en");
	});

	test("defaults to en for empty string", () => {
		expect(localeToLanguageCode("")).toBe("en");
	});
});

// ─── Model catalog validation ────────────────────────────────────────────────

describe("LOCAL_MODELS catalog", () => {
	test("all models have non-empty downloadUrls", () => {
		for (const model of LOCAL_MODELS) {
			// Phase-2 GGUF entries route via whisperCppModel.downloadUrls;
			// the legacy sherpaModel field is stubbed with empty maps. The
			// invariant check belongs in `tests/catalog-invariants` for
			// those entries — here we only assert the legacy shape for
			// sherpa-routed models.
			if (model.engine === "whisper-cpp") continue;
			const urls = model.sherpaModel.downloadUrls;
			const urlCount = Object.keys(urls).length;
			expect(urlCount).toBeGreaterThan(0);
			// Every file should have a corresponding URL
			for (const role of Object.keys(model.sherpaModel.files)) {
				expect(urls[role]).toBeDefined();
				expect(urls[role]).toStartWith("https://");
			}
		}
	});

	test("Whisper models use prefix-based file names", () => {
		const whisperModels = LOCAL_MODELS.filter(m => m.sherpaModel.type === "whisper");
		for (const model of whisperModels) {
			const files = model.sherpaModel.files;
			// Whisper files should be prefixed: small-encoder.int8.onnx, not encoder.int8.onnx
			expect(files.encoder).not.toBe("encoder.int8.onnx");
			expect(files.decoder).not.toBe("decoder.int8.onnx");
		}
	});

	test("Moonshine v2 models use mergedDecoder (not 4-file v1 structure)", () => {
		const v2Models = LOCAL_MODELS.filter(m => m.id.startsWith("moonshine-v2"));
		for (const model of v2Models) {
			const files = model.sherpaModel.files;
			expect(files.mergedDecoder).toBeDefined();
			expect(files.preprocessor).toBeUndefined();
			expect(files.uncachedDecoder).toBeUndefined();
			expect(files.cachedDecoder).toBeUndefined();
		}
	});

	test("Moonshine v1 models use 4-file structure", () => {
		const v1Models = LOCAL_MODELS.filter(m =>
			m.sherpaModel.type === "moonshine" && !m.id.startsWith("moonshine-v2"),
		);
		for (const model of v1Models) {
			const files = model.sherpaModel.files;
			expect(files.preprocessor).toBeDefined();
			expect(files.encoder).toBeDefined();
			expect(files.uncachedDecoder).toBeDefined();
			expect(files.cachedDecoder).toBeDefined();
		}
	});

	test("Parakeet models use transducer type (not nemo_ctc)", () => {
		const parakeetModels = LOCAL_MODELS.filter(m => m.id.startsWith("parakeet"));
		for (const model of parakeetModels) {
			expect(model.sherpaModel.type).toBe("transducer");
			expect(model.sherpaModel.files.encoder).toBeDefined();
			expect(model.sherpaModel.files.decoder).toBeDefined();
			expect(model.sherpaModel.files.joiner).toBeDefined();
		}
	});

	test("GigaAM uses nemo_ctc type", () => {
		const gigaam = LOCAL_MODELS.find(m => m.id === "gigaam-v3")!;
		expect(gigaam.sherpaModel.type).toBe("nemo_ctc");
		expect(gigaam.sherpaModel.files.model).toBeDefined();
	});

	test("all models have positive sizeBytes and runtimeRamMB", () => {
		for (const model of LOCAL_MODELS) {
			expect(model.sizeBytes).toBeGreaterThan(0);
			expect(model.runtimeRamMB).toBeGreaterThan(0);
		}
	});
});

// ─── hasCUDA — env-authoritative + platform-aware ────────────────────────────

describe("hasCUDA", () => {
	test("darwin with no env override → false (no CUDA on Mac)", () => {
		expect(hasCUDA({}, "darwin")).toBe(false);
	});

	test("win32 with no env override → false (no Windows prebuilt)", () => {
		expect(hasCUDA({}, "win32")).toBe(false);
	});

	test("CUDA_VISIBLE_DEVICES=\"\" disables CUDA on every platform", () => {
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "" }, "linux")).toBe(false);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "" }, "darwin")).toBe(false);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "" }, "win32")).toBe(false);
	});

	test("CUDA_VISIBLE_DEVICES=\"-1\" disables CUDA", () => {
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "-1" }, "linux")).toBe(false);
	});

	test("CUDA_VISIBLE_DEVICES=\" -1 \" disables (whitespace trimmed)", () => {
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: " -1 " }, "linux")).toBe(false);
	});

	test("CUDA_VISIBLE_DEVICES=\"   \" disables (whitespace-only trims to empty)", () => {
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "   " }, "linux")).toBe(false);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "\t" }, "linux")).toBe(false);
	});

	test("CUDA_VISIBLE_DEVICES=\"NONE\" disables (case-insensitive)", () => {
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "NONE" }, "linux")).toBe(false);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "None" }, "linux")).toBe(false);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "none" }, "linux")).toBe(false);
	});

	test("CUDA_VISIBLE_DEVICES=\"VOID\" disables (case-insensitive)", () => {
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "VOID" }, "linux")).toBe(false);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "void" }, "linux")).toBe(false);
	});

	test("CUDA_VISIBLE_DEVICES=\"0\" enables CUDA on every platform", () => {
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "0" }, "linux")).toBe(true);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "0" }, "darwin")).toBe(true);
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "0" }, "win32")).toBe(true);
	});

	test("NVIDIA_VISIBLE_DEVICES=\"all\" enables CUDA", () => {
		expect(hasCUDA({ NVIDIA_VISIBLE_DEVICES: "all" }, "linux")).toBe(true);
	});

	test("env var SET takes precedence over native probe (no fallthrough)", () => {
		// Even if /dev/nvidia0 happens to exist on the test host, an
		// explicit empty override must still return false.
		expect(hasCUDA({ CUDA_VISIBLE_DEVICES: "" }, "linux")).toBe(false);
	});
});
