import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
const read = (path: string) => readFileSync(path, "utf8");

describe("bootstrap setup scripts", () => {
	test("ships a macOS bootstrap script with valid bash syntax", () => {
		expect(existsSync("scripts/setup-macos.sh")).toBe(true);
		const script = read("scripts/setup-macos.sh");
		expect(script).toContain("brew install sox");
		expect(script).toContain("pi install npm:@codexstar/pi-listen");
		expect(script).toContain("DEEPGRAM_API_KEY");
		expect(script).toContain("settings.json");
		expect(script).toContain("onboarding");
		expect(script).toContain("daemon.py");
		expect(script).toContain("transcribe.py");
		expect(script).not.toContain('os.environ["BACKEND_NAME"]');
		expect(script).toContain('sys.argv[1]');

		const syntax = spawnSync("bash", ["-n", "scripts/setup-macos.sh"], {
			encoding: "utf8",
			stdio: ["pipe", "pipe", "pipe"],
		});
		expect(syntax.status).toBe(0);
		expect(syntax.stderr).toBe("");
	});

	test("ships a Windows bootstrap script with winget + pi install flow", () => {
		expect(existsSync("scripts/setup-windows.ps1")).toBe(true);
		const script = read("scripts/setup-windows.ps1");
		expect(script).toContain("ChrisBagwell.SoX");
		expect(script).toContain("Python.Python.3.12");
		expect(script).toContain("pi install npm:@codexstar/pi-listen");
		expect(script).toContain("DEEPGRAM_API_KEY");
		expect(script).toContain("settings.json");
		expect(script).toContain("onboarding");
		expect(script).toContain("daemon.py");
		expect(script).toContain("transcribe.py");
	});

	test("documents the setup scripts in the README", () => {
		const readme = read("README.md");
		expect(readme).toContain("scripts/setup-macos.sh");
		expect(readme).toContain("scripts/setup-windows.ps1");
		expect(readme).toContain("You should not need to run `/voice setup` on the happy path");
	});

	test("includes scripts in the published package", () => {
		expect(Array.isArray(packageJson.files)).toBe(true);
		expect(packageJson.files).toContain("scripts");
	});
});
