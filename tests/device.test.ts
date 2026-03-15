import { describe, expect, test } from "bun:test";
import {
	detectDevice,
	getModelFitness,
	autoRecommendModel,
	formatDeviceSummary,
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

	test("model returns warning when runtime exceeds free RAM", () => {
		// whisper-medium needs ~1250 MB, device has only 800 MB free but 4096 total
		const mediumModel = LOCAL_MODELS.find(m => m.id === "whisper-medium")!;
		const device = mockDevice({ totalRamMB: 4096, freeRamMB: 800 });
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
