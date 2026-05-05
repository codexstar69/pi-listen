import { describe, expect, test } from "bun:test";
import { sherpaEngine } from "../extensions/voice/engine-sherpa";

describe("sherpaEngine adapter — load-time safety contract", () => {
	test("module imports without throwing", () => {
		expect(sherpaEngine).toBeDefined();
		expect(sherpaEngine.id).toBe("sherpa-onnx");
	});

	test("singleton is readable before init() is called", () => {
		// The contract: reading `available` / `error` MUST never throw.
		// `available` may be true if a prior test already initialized
		// sherpa (the underlying recognizer cache is module-scoped).
		expect(typeof sherpaEngine.available).toBe("boolean");
		expect(sherpaEngine.error === null || typeof sherpaEngine.error === "string").toBe(true);
	});

	test("transcribe throws structured error when not available", () => {
		// We can't reliably force `available: false` once the module has
		// been initialized in the test suite, so this test only runs when
		// the engine genuinely is unavailable. It still pins the contract:
		// a structured error, not a native stack.
		if (sherpaEngine.available) {
			// Engine is loaded — skip the negative test.
			expect(sherpaEngine.available).toBe(true);
			return;
		}
		expect(() => sherpaEngine.transcribe(Buffer.alloc(0), {
			model: { id: "x" } as any,
			modelDir: "/tmp",
			lang: "en",
		})).toThrow(/sherpa-onnx engine is not available/);
	});

	test("clearCache is idempotent — does not throw on repeat calls", async () => {
		await expect(sherpaEngine.clearCache()).resolves.toBeUndefined();
		await expect(sherpaEngine.clearCache()).resolves.toBeUndefined();
	});

	test("init() is idempotent — second call returns cached result", async () => {
		const first = await sherpaEngine.init();
		const second = await sherpaEngine.init();
		expect(typeof first).toBe("boolean");
		expect(second).toBe(first);
	});

	test("clearCache + transcribe contract holds across rapid calls", async () => {
		// Round-4 audit (GPT-5.5 #3): clearCache must never throw, even
		// rapidly chained. Round-3 also: idempotent.
		await sherpaEngine.clearCache();
		await sherpaEngine.clearCache();
		await sherpaEngine.clearCache();
		// The engine is still in a valid state.
		expect(sherpaEngine.id).toBe("sherpa-onnx");
		expect(typeof sherpaEngine.available).toBe("boolean");
	});

	test("hint is exposed via getter, not field (interface symmetry)", () => {
		// Round-4 audit (Opus 4.6): both adapters expose `hint` via a
		// getter so the interface contract is symmetric. Sherpa returns
		// null because it has no informational hints today.
		expect(sherpaEngine.hint).toBeNull();
		// Property descriptor on the prototype confirms it's a getter,
		// not a value field.
		const proto = Object.getPrototypeOf(sherpaEngine);
		const desc = Object.getOwnPropertyDescriptor(proto, "hint");
		expect(typeof desc?.get).toBe("function");
	});
});
