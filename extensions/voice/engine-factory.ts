/**
 * Engine selection + session-start fallback resolver.
 *
 * Two roles:
 *   1. Map a `LocalModelInfo` to its engine implementation
 *      (`getEngineForModel`). Phase 1 only knows about sherpa; Phase 2 will
 *      add `whisper-cpp` routing.
 *   2. Resolve the effective model id for a session given (possibly stale)
 *      persisted config (`resolveEffectiveModel`). Falls back gracefully
 *      when the persisted id is unsupported on the current runtime,
 *      without mutating persisted config — matches the architect-round-2
 *      portability concern.
 *
 * Catalog invariants:
 *   - `ULTIMATE_FALLBACK_MODEL_ID` MUST be in `LOCAL_MODELS`.
 *   - It MUST have no `supports` constraint (universally selectable).
 *   - It MUST be a sherpa-onnx model (so `PI_VOICE_FORCE_SHERPA=1` always
 *     resolves).
 * These are enforced lazily by `ensureCatalogValid()` (first-call) AND
 * up-front by `tests/catalog-invariants.test.ts` at CI time. The CI gate
 * is the real defense; the lazy check is the runtime backstop for hand-
 * edited installs.
 *
 * Why lazy and not top-level: `local.ts` imports from this module, and
 * this module imports `LOCAL_MODELS` from `local.ts` — a mutual import. A
 * top-level read of `LOCAL_MODELS` here hits the ESM TDZ window in
 * whichever order the loader picks. Deferring the read to first call
 * breaks the trap.
 */

import { LOCAL_MODELS, type LocalModelInfo } from "./local";
import {
	currentVoiceRuntimeEnv,
	isModelSupported,
	type VoiceRuntimeEnv,
} from "./model-compat";
import {
	VoiceCatalogUnrecoverableError,
	type Engine,
	type EngineId,
} from "./engine";
import { sherpaEngine } from "./engine-sherpa";
import { whispercppEngine } from "./engine-whispercpp";

// ─── Constants ───────────────────────────────────────────────────────────────

/**
 * The single id guaranteed to work on every supported runtime. Picked for:
 * small (~43 MB), no `supports` constraint, fast on CPU, English-only but
 * acceptable as a compatibility fallback. Exported so tests and the
 * `catalog-invariants` suite reference the same constant.
 */
export const ULTIMATE_FALLBACK_MODEL_ID = "moonshine-v2-tiny";

/**
 * Default models in priority order. Fresh installs scan top-to-bottom and
 * pick the first id that's both in the catalog AND `isModelSupported` on
 * the current runtime. Phase 1 ships with only sherpa-onnx entries
 * present; Phase 2 adds the whisper.cpp ids at the top.
 *
 * Entries may reference ids not yet in the catalog (during incremental
 * rollout). Missing ids are skipped, NOT treated as wildcards.
 */
const DEFAULT_MODEL_PRIORITY: readonly string[] = [
	"whisper-base-en-metal",          // 57 MB whisper.cpp Metal — Phase 2
	"whisper-large-v3-turbo-metal",   // 800 MB Turbo — Phase 2 opt-in upgrade
	"whisper-large-v3-turbo-cuda",    // Linux CUDA — Phase 3
	ULTIMATE_FALLBACK_MODEL_ID,       // sherpa-onnx Moonshine v2 Tiny — Phase 1
];

// ─── Catalog invariants (lazy) ───────────────────────────────────────────────

let _catalogValidated = false;
/** Lazily-built id → entry map. Round-2 audit perf fix: O(1) lookups
 *  in the resolver vs N×O(n) `LOCAL_MODELS.find()` calls per session
 *  start. Rebuilt by `__resetCatalogValidationForTest()`. */
let _modelMap: Map<string, LocalModelInfo> | null = null;

function getModelMap(): Map<string, LocalModelInfo> {
	if (!_modelMap) _modelMap = new Map(LOCAL_MODELS.map(m => [m.id, m]));
	return _modelMap;
}

function findModelById(id: string): LocalModelInfo | undefined {
	return getModelMap().get(id);
}

/**
 * Lazy invariant check — runs once on first export call. Reading
 * `LOCAL_MODELS` only inside this function avoids the circular-import TDZ
 * trap that a top-level read would create.
 */
function ensureCatalogValid(): void {
	if (_catalogValidated) return;
	const m = findModelById(ULTIMATE_FALLBACK_MODEL_ID);
	if (!m) {
		throw new Error(
			`Catalog is missing the ultimate fallback model ` +
			`"${ULTIMATE_FALLBACK_MODEL_ID}" — check extensions/voice/local.ts.`,
		);
	}
	if (m.supports && m.supports.length > 0) {
		throw new Error(
			`Ultimate fallback "${ULTIMATE_FALLBACK_MODEL_ID}" must be ` +
			`platform-agnostic (no \`supports\` constraint).`,
		);
	}
	if (m.engine !== undefined && m.engine !== "sherpa-onnx") {
		throw new Error(
			`Ultimate fallback "${ULTIMATE_FALLBACK_MODEL_ID}" must be a ` +
			`sherpa-onnx model so PI_VOICE_FORCE_SHERPA=1 always resolves; ` +
			`got engine="${m.engine}".`,
		);
	}
	_catalogValidated = true;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * True when the user has set `PI_VOICE_FORCE_SHERPA=1`, asking the picker
 * / resolver to skip whisper.cpp models even when supported. Consulted as
 * a function so tests can stub `process.env`.
 *
 * The `config.voiceEngine` setting (UI-level engine preference) maps to
 * the same gate: "sherpa-onnx" → force sherpa, "whisper-cpp" → force
 * whisper-cpp, "auto"/undefined → no force. See `pickDefaultModelForPlatform`
 * for how the preference is wired in.
 */
function forceSherpa(env: NodeJS.ProcessEnv = process.env): boolean {
	return env.PI_VOICE_FORCE_SHERPA === "1";
}

/** Decode the `voiceEngine` config into a force flag. Tests stub via
 *  the optional arg. */
function shouldForceSherpa(
	preference: "auto" | "sherpa-onnx" | "whisper-cpp" | undefined,
	env: NodeJS.ProcessEnv = process.env,
): boolean {
	if (preference === "sherpa-onnx") return true;
	if (preference === "whisper-cpp") return false;
	return forceSherpa(env);
}

function shouldForceWhisperCpp(
	preference: "auto" | "sherpa-onnx" | "whisper-cpp" | undefined,
): boolean {
	return preference === "whisper-cpp";
}

/**
 * Treat a model as a sherpa-onnx model when its `engine` is undefined OR
 * explicitly `"sherpa-onnx"`. Matches `getEngineForModel`'s default — the
 * undefined case is the back-compat path for catalog entries that pre-date
 * the field.
 */
function isSherpaModel(model: LocalModelInfo): boolean {
	return model.engine === undefined || model.engine === "sherpa-onnx";
}

/**
 * Find an id likely (but not guaranteed) to exist in `LOCAL_MODELS`,
 * walking four tiers of preference:
 *   1. `ULTIMATE_FALLBACK_MODEL_ID` if it's in the catalog.
 *   2. Any platform-supported sherpa-onnx model (Codex round-15 polish —
 *      prefer something that will actually load on the current runtime
 *      before reaching for a platform-locked last-resort).
 *   3. Any platform-agnostic sherpa-onnx model.
 *   4. `LOCAL_MODELS[0]` last resort — may be platform-locked, but exists,
 *      so transcribe gets a descriptive engine error rather than the
 *      cryptic "Unknown model: …".
 * Returns `null` only when `LOCAL_MODELS` is empty — at which point the
 * install is genuinely unrecoverable and the caller throws
 * `VoiceCatalogUnrecoverableError`.
 */
function findResilientFallbackId(
	env: VoiceRuntimeEnv,
	processEnv: NodeJS.ProcessEnv = process.env,
): string | null {
	const ultimate = findModelById(ULTIMATE_FALLBACK_MODEL_ID);
	if (ultimate) return ULTIMATE_FALLBACK_MODEL_ID;

	const supportedSherpa = LOCAL_MODELS.find(m =>
		isSherpaModel(m) && isModelSupported(m, env),
	);
	if (supportedSherpa) return supportedSherpa.id;

	const agnosticSherpa = LOCAL_MODELS.find(m =>
		isSherpaModel(m) && (!m.supports || m.supports.length === 0),
	);
	if (agnosticSherpa) return agnosticSherpa.id;

	// Tier 4 — last-resort. Round-2 audit polish: when
	// PI_VOICE_FORCE_SHERPA=1 the user explicitly opted out of whisper.cpp;
	// prefer ANY sherpa entry over a whisper-cpp last-resort even if both
	// platform-locked. Falls back to LOCAL_MODELS[0] only when no sherpa
	// model exists at all (truly broken catalog).
	const anySherpa = LOCAL_MODELS.find(isSherpaModel);
	if (anySherpa) return anySherpa.id;

	const skipWhisperCpp = forceSherpa(processEnv);
	if (!skipWhisperCpp && LOCAL_MODELS.length > 0 && LOCAL_MODELS[0]) {
		return LOCAL_MODELS[0].id;
	}
	return null;
}

/**
 * Build the degraded `resolveEffectiveModel` result given catalog
 * corruption. Throws `VoiceCatalogUnrecoverableError` if the catalog is
 * empty (genuinely unrecoverable). One implementation, two call sites
 * (the outer `ensureCatalogValid` catch and the inner `safeDefault`
 * catch) — keeps fallback policy in a single location.
 */
function degradedResolve(
	persistedId: string | undefined,
	env: VoiceRuntimeEnv,
	rootCause: unknown,
	processEnv: NodeJS.ProcessEnv = process.env,
): { id: string; fellBack: boolean; from?: string; degraded: true } {
	const id = findResilientFallbackId(env, processEnv);
	if (id === null) {
		throw new VoiceCatalogUnrecoverableError(
			`Catalog invariants failed and LOCAL_MODELS is empty — ` +
			`${String(rootCause)}. Reinstall pi-voice or check for a ` +
			`corrupted extension build.`,
		);
	}
	// eslint-disable-next-line no-console
	console.error(
		`[pi-voice] resolveEffectiveModel: catalog invariants failed — ` +
		`${String(rootCause)}. Falling back to "${id}".`,
	);
	return {
		id,
		fellBack: persistedId !== undefined && persistedId !== id,
		from: persistedId,
		degraded: true,
	};
}

// ─── Engine selection ────────────────────────────────────────────────────────

/** Map a model to its engine. `model.engine` undefined → sherpa-onnx
 *  (back-compat for catalog entries that pre-date the field).
 *
 *  Round-2 audit polish: explicit throw on unknown engine values so a
 *  catalog-corruption-induced typo surfaces with diagnostic context
 *  instead of silently routing to sherpa with later confusing errors.
 *  TypeScript prevents this in source, but a hand-edited config or
 *  third-party catalog can defeat the type system at runtime. */
export function getEngineForModel(model: LocalModelInfo): Engine {
	if (model.engine === undefined || model.engine === "sherpa-onnx") return sherpaEngine;
	if (model.engine === "whisper-cpp") return whispercppEngine;
	throw new Error(
		`Unknown STT engine "${(model as { engine: unknown }).engine}" for ` +
		`model "${model.id}". Expected "sherpa-onnx" or "whisper-cpp" or undefined.`,
	);
}

// ─── Default resolution ─────────────────────────────────────────────────────

export function pickDefaultModelForPlatform(
	env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
	processEnv: NodeJS.ProcessEnv = process.env,
	enginePreference?: "auto" | "sherpa-onnx" | "whisper-cpp",
): string {
	ensureCatalogValid();
	const skipWhisperCpp = shouldForceSherpa(enginePreference, processEnv);
	const skipSherpa = shouldForceWhisperCpp(enginePreference);
	for (const id of DEFAULT_MODEL_PRIORITY) {
		const model = findModelById(id);
		if (!model) continue;                       // skip un-shipped IDs
		if (skipWhisperCpp && !isSherpaModel(model)) continue;
		if (skipSherpa && isSherpaModel(model)) continue;
		if (isModelSupported(model, env)) return id;
	}
	// `ensureCatalogValid()` guarantees the ultimate fallback is in
	// LOCAL_MODELS with no `supports` constraint AND is a sherpa model
	// (so `forceSherpa` doesn't filter it out), so the loop above MUST
	// have returned. Reaching this line means the catalog was mutated
	// mid-iteration or the validator was bypassed — fail loudly with
	// diagnostic context, never silently substitute.
	throw new Error(
		`pickDefaultModelForPlatform: no supported model found in catalog ` +
		`for ${env.platform}-${env.arch} (hasCUDA=${env.hasCUDA}, ` +
		`forceSherpa=${skipWhisperCpp}). The ultimate fallback ` +
		`"${ULTIMATE_FALLBACK_MODEL_ID}" should always match — this is a ` +
		`catalog invariant violation.`,
	);
}

/**
 * Resolve the effective model for a session, given (possibly stale)
 * persisted config. Falls back to the platform default when:
 *   - persisted id isn't in the catalog, OR
 *   - persisted id isn't supported on the current runtime, OR
 *   - PI_VOICE_FORCE_SHERPA=1 and the persisted model is whisper-cpp.
 *
 * Returns `{ fellBack: true, from }` so the caller can surface a notice
 * via `migratePersistedModelOnSessionStart`. `degraded: true` indicates
 * catalog corruption — caller should show a louder warning. Throws
 * `VoiceCatalogUnrecoverableError` only when the catalog is empty.
 */
export function resolveEffectiveModel(
	persistedId: string | undefined,
	env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
	processEnv: NodeJS.ProcessEnv = process.env,
	enginePreference?: "auto" | "sherpa-onnx" | "whisper-cpp",
): { id: string; fellBack: boolean; from?: string; degraded?: boolean } {
	// `ensureCatalogValid()` can throw on a corrupted catalog. Wrap so
	// session_start still gets a working session when possible.
	try {
		ensureCatalogValid();
	} catch (err) {
		return degradedResolve(persistedId, env, err, processEnv);
	}

	const skipWhisperCpp = shouldForceSherpa(enginePreference, processEnv);
	const skipSherpa = shouldForceWhisperCpp(enginePreference);

	function safeDefault(): { id: string; degraded: boolean } {
		try {
			return { id: pickDefaultModelForPlatform(env, processEnv, enginePreference), degraded: false };
		} catch (err) {
			// Round-6 audit (GPT-5.5 Amp BLOCK #3): when the user
			// explicitly forced whisper-cpp via `enginePreference` and no
			// whisper-cpp model is supported on this runtime, the
			// resilient fallback would silently substitute a sherpa-onnx
			// model — exact opposite of what the user asked for. Surface
			// the error instead so the UI can show "no whisper-cpp model
			// supported on this platform; revert your engine preference"
			// rather than silently routing to a sherpa entry under a UI
			// labelled "whisper.cpp forced".
			if (shouldForceWhisperCpp(enginePreference)) throw err;
			// `pickDefault` should never throw if `ensureCatalogValid`
			// above succeeded — defense-in-depth against any future
			// invariant we forget to enforce in the validator. Re-use the
			// shared fallback search.
			const id = findResilientFallbackId(env, processEnv);
			if (id === null) {
				throw new VoiceCatalogUnrecoverableError(
					`pickDefaultModelForPlatform threw and no resilient fallback ` +
					`found in LOCAL_MODELS — ${String(err)}`,
				);
			}
			// eslint-disable-next-line no-console
			console.error(
				`[pi-voice] resolveEffectiveModel: pickDefault threw — ${String(err)}. ` +
				`Falling back to "${id}".`,
			);
			return { id, degraded: true };
		}
	}

	if (persistedId) {
		const m = findModelById(persistedId);
		const blockedByForceSherpa = m && skipWhisperCpp && !isSherpaModel(m);
		const blockedByForceWhisperCpp = m && skipSherpa && isSherpaModel(m);
		if (m && isModelSupported(m, env) && !blockedByForceSherpa && !blockedByForceWhisperCpp) {
			return { id: persistedId, fellBack: false };
		}
		const def = safeDefault();
		return { id: def.id, fellBack: true, from: persistedId, degraded: def.degraded };
	}
	const def = safeDefault();
	return { id: def.id, fellBack: false, degraded: def.degraded };
}

// ─── Test seam ───────────────────────────────────────────────────────────────

/**
 * Reset the lazy catalog-validation cache. Test-only — call in a
 * `beforeEach` when injecting a corrupted catalog mock so the validator
 * AND the lookup map both re-run against the new state.
 */
export function __resetCatalogValidationForTest(): void {
	_catalogValidated = false;
	_modelMap = null;
}
