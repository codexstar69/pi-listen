import { spawn } from "node:child_process";
import type { VoiceConfig } from "./config";
import type { EnvironmentDiagnostics } from "./diagnostics";

export interface ProvisioningPlan {
	ready: boolean;
	summary: string;
	commands: string[];
	manualSteps: string[];
}

export interface ProvisioningResult {
	success: boolean;
	ran: string[];
	failed: { command: string; error: string }[];
	skipped: string[];
}

/**
 * Execute all auto-runnable install commands from a provisioning plan.
 * Runs each command sequentially, reports results.
 * Manual steps are returned as skipped.
 */
export async function executeProvisioningPlan(
	plan: ProvisioningPlan,
	onProgress?: (message: string) => void,
): Promise<ProvisioningResult> {
	const ran: string[] = [];
	const failed: { command: string; error: string }[] = [];

	for (const cmd of plan.commands) {
		onProgress?.(`Running: ${cmd}`);
		try {
			await runShellCommand(cmd);
			ran.push(cmd);
			onProgress?.(`✓ ${cmd}`);
		} catch (err: any) {
			const errMsg = err?.message || String(err);
			failed.push({ command: cmd, error: errMsg });
			onProgress?.(`✗ ${cmd}: ${errMsg}`);
		}
	}

	return {
		success: failed.length === 0 && plan.manualSteps.length === 0,
		ran,
		failed,
		skipped: plan.manualSteps,
	};
}

function runShellCommand(cmd: string, timeoutMs = 300_000): Promise<string> {
	return new Promise((resolve, reject) => {
		const proc = spawn("sh", ["-c", cmd], {
			stdio: ["pipe", "pipe", "pipe"],
			env: { ...process.env, PIP_BREAK_SYSTEM_PACKAGES: "1" },
		});

		let stdout = "";
		let stderr = "";
		proc.stdout?.on("data", (d: Buffer) => { stdout += d.toString(); });
		proc.stderr?.on("data", (d: Buffer) => { stderr += d.toString(); });

		const timer = setTimeout(() => {
			try { proc.kill("SIGTERM"); } catch {}
			reject(new Error(`Timed out after ${timeoutMs / 1000}s`));
		}, timeoutMs);

		proc.on("close", (code) => {
			clearTimeout(timer);
			if (code === 0) resolve(stdout);
			else reject(new Error(stderr.slice(-500) || `Exit code ${code}`));
		});
		proc.on("error", (err) => {
			clearTimeout(timer);
			reject(err);
		});
	});
}

export function buildProvisioningPlan(config: VoiceConfig, diagnostics: EnvironmentDiagnostics): ProvisioningPlan {
	const commands: string[] = [];
	const manualSteps: string[] = [];

	if (!diagnostics.hasSox) {
		commands.push("brew install sox");
	}

	if (config.mode === "api") {
		if (config.backend === "deepgram" && !diagnostics.hasDeepgramKey) {
			manualSteps.push("Set DEEPGRAM_API_KEY — get a free key at https://dpgr.am/pi-voice ($200 free credits)");
		}
	} else {
		if (!diagnostics.hasPython) {
			manualSteps.push("Install python3 before enabling local STT backends");
		}

		const selectedBackend = diagnostics.backends.find((backend) => backend.name === config.backend);
		if (!selectedBackend?.available) {
			switch (config.backend) {
				case "faster-whisper":
					commands.push("python3 -m pip install faster-whisper");
					break;
				case "moonshine":
					commands.push("python3 -m pip install 'useful-moonshine[onnx]'");
					break;
				case "whisper-cpp":
					commands.push("brew install whisper-cpp");
					break;
				case "parakeet":
					commands.push("python3 -m pip install 'nemo_toolkit[asr]'");
					break;
				case "parakeet-mlx":
					commands.push("pip install parakeet-mlx");
					break;
				default:
					manualSteps.push(`Install the selected backend: ${config.backend}`);
			}
		} else if (!(selectedBackend.installed_models ?? []).includes(config.model)) {
			const highConfidenceDetectors = new Set(["huggingface-cache", "whisper-cpp-model-paths"]);
			if (highConfidenceDetectors.has(selectedBackend.install_detection ?? "")) {
				manualSteps.push(`Selected model ${config.model} is not installed yet and may need to be downloaded on first use`);
			} else {
				manualSteps.push(`Selected model ${config.model} could not be confirmed locally and may still need a download on first use`);
			}
		}
	}

	return {
		ready: commands.length === 0 && manualSteps.length === 0,
		summary: config.mode === "api" ? "Provisioning plan for api mode" : "Provisioning plan for local mode",
		commands,
		manualSteps,
	};
}
