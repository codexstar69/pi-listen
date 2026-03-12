import { describe, expect, test } from "bun:test";
import { type VoiceConfig, DEFAULT_CONFIG } from "../extensions/voice/config";
import { buildProvisioningPlan } from "../extensions/voice/install";
import type { EnvironmentDiagnostics } from "../extensions/voice/diagnostics";

function makeDiagnostics(overrides: Partial<EnvironmentDiagnostics> = {}): EnvironmentDiagnostics {
	return {
		hasPython: true,
		hasSox: true,
		hasHomebrew: true,
		hasDeepgramKey: false,
		backends: [
			{ name: "faster-whisper", available: false, type: "local", default_model: "small", models: ["small"], install: "pip install faster-whisper" },
			{ name: "moonshine", available: false, type: "local", default_model: "moonshine/base", models: ["moonshine/base"], install: "pip install useful-moonshine[onnx]" },
			{ name: "whisper-cpp", available: false, type: "local", default_model: "small", models: ["small"], install: "brew install whisper-cpp" },
			{ name: "deepgram", available: false, type: "cloud", default_model: "nova-3", models: ["nova-3"], install: "Set DEEPGRAM_API_KEY" },
			{ name: "parakeet", available: false, type: "local", default_model: "nvidia/parakeet-tdt-0.6b-v2", models: ["nvidia/parakeet-tdt-0.6b-v2"], install: "pip install nemo_toolkit[asr]" },
		],
		issues: [],
		...overrides,
	};
}

function makeConfig(overrides: Partial<VoiceConfig> = {}): VoiceConfig {
	return {
		...DEFAULT_CONFIG,
		mode: "local",
		backend: "faster-whisper",
		model: "small",
		onboarding: {
			completed: true,
			schemaVersion: DEFAULT_CONFIG.version,
		},
		...overrides,
	};
}

describe("buildProvisioningPlan", () => {
	test("includes SoX and backend install commands for missing local dependencies", () => {
		const diagnostics = makeDiagnostics({ hasSox: false });
		const config = makeConfig();

		const plan = buildProvisioningPlan(config, diagnostics);

		expect(plan.summary).toContain("local");
		expect(plan.commands).toContain("brew install sox");
		expect(plan.commands).toContain("python3 -m pip install faster-whisper");
	});

	test("flags missing Deepgram key for api mode", () => {
		const diagnostics = makeDiagnostics();
		const config = makeConfig({ mode: "api", backend: "deepgram", model: "nova-3" });

		const plan = buildProvisioningPlan(config, diagnostics);

		expect(plan.manualSteps).toContain("Set DEEPGRAM_API_KEY — get a free key at https://dpgr.am/pi-voice ($200 free credits)");
	});

	test("returns ready state when dependencies already exist", () => {
		const diagnostics = makeDiagnostics({
			hasDeepgramKey: true,
			hasSox: true,
			backends: makeDiagnostics().backends.map((backend) =>
				backend.name === "faster-whisper" ? { ...backend, available: true, installed_models: ["small"] } : backend,
			),
		});
		const config = makeConfig();

		const plan = buildProvisioningPlan(config, diagnostics);

		expect(plan.commands.length).toBe(0);
		expect(plan.manualSteps.length).toBe(0);
		expect(plan.ready).toBe(true);
	});

	test("surfaces when a selected local model is not yet installed", () => {
		const diagnostics = makeDiagnostics({
			backends: makeDiagnostics().backends.map((backend) =>
				backend.name === "faster-whisper"
					? { ...backend, available: true, installed_models: ["small"], install_detection: "huggingface-cache" }
					: backend,
			),
		});
		const config = makeConfig({ model: "medium" });

		const plan = buildProvisioningPlan(config, diagnostics);

		expect(plan.ready).toBe(false);
		expect(plan.manualSteps).toContain("Selected model medium is not installed yet and may need to be downloaded on first use");
	});

	test("uses conservative wording when model presence cannot be confirmed with high confidence", () => {
		const diagnostics = makeDiagnostics({
			backends: makeDiagnostics().backends.map((backend) =>
				backend.name === "parakeet"
					? { ...backend, available: true, installed_models: [], install_detection: "huggingface-or-nemo-cache" }
					: backend,
			),
		});
		const config = makeConfig({ backend: "parakeet", model: "nvidia/parakeet-tdt-0.6b-v2" });

		const plan = buildProvisioningPlan(config, diagnostics);

		expect(plan.ready).toBe(false);
		expect(plan.manualSteps).toContain("Selected model nvidia/parakeet-tdt-0.6b-v2 could not be confirmed locally and may still need a download on first use");
	});
});
