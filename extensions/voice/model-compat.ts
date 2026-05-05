/**
 * Model compatibility — runtime/platform/CUDA support resolution.
 *
 * Centralised so every consumer (settings filter, default picker, download
 * guard, runtime activation) calls through the same predicates. The contract
 * lives in this module; everywhere else just calls `isModelSupported(model)`.
 *
 * `assertModelSupported` is deliberately NOT exported. Hard-failing on
 * activation makes persisted config non-portable across machines, which the
 * round-2 architect review flagged as a real migration hazard. Settings UI
 * filters unsupported models out, and `engine-factory.resolveEffectiveModel`
 * does graceful session-start fallback. The session_start flow is the only
 * place that can resolve a stale persisted id, and it never overwrites the
 * persisted config.
 */

import { cudaBuildCapable, hasCUDA } from "./device";
import type { LocalModelInfo } from "./local";

// ─── Types ───────────────────────────────────────────────────────────────────

/**
 * Per-model platform/arch/GPU support. A model is considered supported when
 * AT LEAST ONE entry in its `supports` array fully matches the runtime env.
 *
 * `arch` undefined → any arch on the named platform.
 * `requiresCUDA` true → requires `env.hasCUDA`. Undefined / false → no
 * requirement.
 */
export interface ModelSupport {
	platform: NodeJS.Platform;
	arch?: NodeJS.Architecture;
	requiresCUDA?: boolean;
}

export interface VoiceRuntimeEnv {
	platform: NodeJS.Platform;
	arch: NodeJS.Architecture;
	hasCUDA: boolean;
	/** True when `smart-whisper` was built with CUDA support (signaled
	 *  via `WHISPER_CUDA=1` in the runtime env). Distinct from `hasCUDA`,
	 *  which only knows about the system's GPU/driver presence. A
	 *  `requiresCUDA: true` model needs BOTH to be selectable. */
	cudaBuildCapable: boolean;
}

// ─── Runtime env ─────────────────────────────────────────────────────────────

/** Snapshot the current runtime environment. Reads `process.platform` /
 *  `process.arch` and consults `hasCUDA()` once. Pure — no caching. */
export function currentVoiceRuntimeEnv(): VoiceRuntimeEnv {
	return {
		platform: process.platform,
		arch: process.arch,
		hasCUDA: hasCUDA(),
		cudaBuildCapable: cudaBuildCapable(),
	};
}

// ─── Predicates ──────────────────────────────────────────────────────────────

/**
 * A single ModelSupport row matches the runtime if every constraint it
 * specifies is satisfied. The three checks are independent — arch and CUDA
 * validation are separate gates and neither is bypassed by the other.
 */
function matchesSupport(s: ModelSupport, env: VoiceRuntimeEnv): boolean {
	if (s.platform !== env.platform) return false;
	if (s.arch !== undefined && s.arch !== env.arch) return false;
	// `requiresCUDA: true` needs BOTH the system to have CUDA hardware
	// AND the smart-whisper binding to have been built with CUDA support.
	// See `device.ts:cudaBuildCapable` for why both are necessary.
	if (s.requiresCUDA === true && (!env.hasCUDA || !env.cudaBuildCapable)) return false;
	return true;
}

/**
 * Whether `model` is supported on the given runtime env. A model with no
 * `supports` array (or an empty one) is considered universally supported —
 * matches the back-compat default for legacy sherpa-onnx catalog entries
 * that pre-date the field.
 */
export function isModelSupported(
	model: LocalModelInfo,
	env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
): boolean {
	if (!model.supports || model.supports.length === 0) return true;
	return model.supports.some(s => matchesSupport(s, env));
}

/**
 * Human-readable reason a model is unsupported on the given env. Used in
 * notifications + error messages, never thrown.
 */
export function describeUnsupported(
	model: LocalModelInfo,
	env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
): string {
	const arch = `${env.platform}-${env.arch}`;
	const needsCUDA = model.supports?.some(s => s.requiresCUDA) ?? false;
	let suffix = "";
	if (needsCUDA && !env.hasCUDA) {
		suffix = " (CUDA hardware unavailable)";
	} else if (needsCUDA && !env.cudaBuildCapable) {
		suffix = " (smart-whisper built without CUDA — reinstall with WHISPER_CUDA=1)";
	}
	return `Model ${model.id} is not supported on ${arch}${suffix}`;
}
