import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
	__resetCatalogValidationForTest,
	getEngineForModel,
	pickDefaultModelForPlatform,
	resolveEffectiveModel,
	ULTIMATE_FALLBACK_MODEL_ID,
} from "../extensions/voice/engine-factory";
import { LOCAL_MODELS } from "../extensions/voice/local";
import type { LocalModelInfo } from "../extensions/voice/local";
import { sherpaEngine } from "../extensions/voice/engine-sherpa";
import { VoiceCatalogUnrecoverableError } from "../extensions/voice/engine";

const mac = { platform: "darwin" as const, arch: "arm64" as const, hasCUDA: false, cudaBuildCapable: false };
const linux = { platform: "linux" as const, arch: "x64" as const, hasCUDA: false, cudaBuildCapable: false };
// Linux + CUDA hardware + build-capable smart-whisper. requiresCUDA gating
// needs both flags now (round-2 audit).
const linuxCuda = { platform: "linux" as const, arch: "x64" as const, hasCUDA: true, cudaBuildCapable: true };

beforeEach(() => __resetCatalogValidationForTest());

describe("getEngineForModel", () => {
	test("undefined engine routes to sherpa-onnx (back-compat)", () => {
		const sherpaOnly = LOCAL_MODELS.find(m => m.engine === undefined);
		expect(sherpaOnly).toBeDefined();
		expect(getEngineForModel(sherpaOnly!).id).toBe("sherpa-onnx");
	});

	test("engine: sherpa-onnx routes to sherpa singleton", () => {
		const sherpaOnly = LOCAL_MODELS.find(m => m.engine === undefined);
		const m: LocalModelInfo = { ...sherpaOnly!, engine: "sherpa-onnx" };
		expect(getEngineForModel(m)).toBe(sherpaEngine);
	});

	test("engine: whisper-cpp routes to whisper-cpp singleton (Phase 2)", () => {
		const wc = LOCAL_MODELS.find(m => m.engine === "whisper-cpp");
		expect(wc).toBeDefined();
		expect(getEngineForModel(wc!).id).toBe("whisper-cpp");
	});

	test("unknown engine value throws (round-2 audit defense-in-depth)", () => {
		// TS type prevents this in source; runtime guard catches catalog
		// corruption / hand-edited config that defeats the type system.
		const m = { ...LOCAL_MODELS[0]!, engine: "bogus-engine" } as any;
		expect(() => getEngineForModel(m)).toThrow(/Unknown STT engine "bogus-engine"/);
	});
});

describe("ensureCatalogValid (lazy)", () => {
	test("passes against the real catalog (ultimate fallback present + sherpa + agnostic)", () => {
		// Real LOCAL_MODELS — should satisfy all three invariants.
		expect(() => pickDefaultModelForPlatform(mac)).not.toThrow();
	});
});

describe("pickDefaultModelForPlatform", () => {
	test("Apple Silicon → whisper-base-en-metal (Phase 2 default)", () => {
		expect(pickDefaultModelForPlatform(mac)).toBe("whisper-base-en-metal");
	});

	test("Linux + CUDA → whisper-large-v3-turbo-cuda (Phase 3)", () => {
		// DEFAULT_MODEL_PRIORITY scans Metal first (skipped on Linux), then
		// CUDA, then ULTIMATE_FALLBACK. CUDA matches when env.hasCUDA.
		expect(pickDefaultModelForPlatform(linuxCuda)).toBe("whisper-large-v3-turbo-cuda");
	});

	test("Linux without CUDA → ultimate fallback (sherpa-onnx)", () => {
		expect(pickDefaultModelForPlatform(linux)).toBe(ULTIMATE_FALLBACK_MODEL_ID);
	});

	test("Apple Silicon + PI_VOICE_FORCE_SHERPA=1 → ultimate fallback", () => {
		const id = pickDefaultModelForPlatform(mac, { PI_VOICE_FORCE_SHERPA: "1" });
		expect(id).toBe(ULTIMATE_FALLBACK_MODEL_ID);
	});
});

describe("resolveEffectiveModel", () => {
	test("returns persisted id unchanged when supported (sherpa)", () => {
		const result = resolveEffectiveModel(ULTIMATE_FALLBACK_MODEL_ID, mac);
		expect(result.id).toBe(ULTIMATE_FALLBACK_MODEL_ID);
		expect(result.fellBack).toBe(false);
	});

	test("returns whisper-cpp persisted id unchanged on supported runtime", () => {
		const result = resolveEffectiveModel("whisper-base-en-metal", mac);
		expect(result.id).toBe("whisper-base-en-metal");
		expect(result.fellBack).toBe(false);
	});

	test("undefined persisted id → platform default (Phase 2: Metal on M-series)", () => {
		const result = resolveEffectiveModel(undefined, mac);
		expect(result.id).toBe("whisper-base-en-metal");
		expect(result.fellBack).toBe(false);
	});

	test("unknown persisted id → fellBack=true", () => {
		const result = resolveEffectiveModel("does-not-exist", mac);
		expect(result.fellBack).toBe(true);
		expect(result.from).toBe("does-not-exist");
	});

	test("Metal-only persisted id on Linux → fellBack to sherpa default", () => {
		const result = resolveEffectiveModel("whisper-base-en-metal", linux);
		expect(result.fellBack).toBe(true);
		expect(result.from).toBe("whisper-base-en-metal");
		expect(result.id).toBe(ULTIMATE_FALLBACK_MODEL_ID);
	});

	test("PI_VOICE_FORCE_SHERPA=1 → falls a whisper-cpp persisted id back to sherpa", () => {
		const result = resolveEffectiveModel(
			"whisper-base-en-metal",
			mac,
			{ PI_VOICE_FORCE_SHERPA: "1" },
		);
		expect(result.fellBack).toBe(true);
		expect(result.from).toBe("whisper-base-en-metal");
		expect(result.id).toBe(ULTIMATE_FALLBACK_MODEL_ID);
	});

	test("PI_VOICE_FORCE_SHERPA=1 → leaves a sherpa persisted id unchanged", () => {
		const result = resolveEffectiveModel(
			ULTIMATE_FALLBACK_MODEL_ID,
			mac,
			{ PI_VOICE_FORCE_SHERPA: "1" },
		);
		expect(result.id).toBe(ULTIMATE_FALLBACK_MODEL_ID);
		expect(result.fellBack).toBe(false);
	});
});

// Catalog-corruption simulation: bypass invariants by mutating a runtime
// copy of LOCAL_MODELS. Tests share the spliced state via beforeEach reset.
describe("resolveEffectiveModel — degraded paths", () => {
	const snapshot: LocalModelInfo[] = [];

	beforeEach(() => {
		snapshot.push(...LOCAL_MODELS);
	});

	afterEach(() => {
		// Restore catalog so subsequent tests see real models.
		LOCAL_MODELS.length = 0;
		LOCAL_MODELS.push(...snapshot);
		snapshot.length = 0;
		__resetCatalogValidationForTest();
	});

	test("missing ultimate fallback → degraded:true with surrogate id", () => {
		// Remove the fallback. Validator throws. degradedResolve searches
		// for a sherpa replacement.
		const idx = LOCAL_MODELS.findIndex(m => m.id === ULTIMATE_FALLBACK_MODEL_ID);
		if (idx >= 0) LOCAL_MODELS.splice(idx, 1);

		const result = resolveEffectiveModel(undefined, mac);
		expect(result.degraded).toBe(true);
		expect(result.id).not.toBe(ULTIMATE_FALLBACK_MODEL_ID);
		expect(LOCAL_MODELS.find(m => m.id === result.id)).toBeDefined();
	});

	test("empty catalog → throws VoiceCatalogUnrecoverableError", () => {
		LOCAL_MODELS.length = 0;
		expect(() => resolveEffectiveModel(undefined, mac))
			.toThrow(VoiceCatalogUnrecoverableError);
	});

	test("Codex polish — degraded fallback prefers a runtime-supported sherpa model", () => {
		// Remove the platform-agnostic fallback. Replace with two sherpa
		// entries: one platform-locked to a non-matching env, one supported
		// on `mac`. The degraded resolver must pick the supported one
		// before reaching for the platform-locked LOCAL_MODELS[0].
		LOCAL_MODELS.length = 0;
		LOCAL_MODELS.push({
			...snapshot[0]!,
			id: "linux-only-sherpa",
			supports: [{ platform: "linux" }],
			engine: "sherpa-onnx",
		});
		LOCAL_MODELS.push({
			...snapshot[0]!,
			id: "darwin-supported-sherpa",
			supports: [{ platform: "darwin" }],
			engine: "sherpa-onnx",
		});

		const result = resolveEffectiveModel(undefined, mac);
		expect(result.degraded).toBe(true);
		expect(result.id).toBe("darwin-supported-sherpa");
	});
});
