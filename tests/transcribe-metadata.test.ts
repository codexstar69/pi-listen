import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import * as path from "node:path";

const SCRIPT_PATH = path.join(process.cwd(), "transcribe.py");

describe("transcribe metadata", () => {
	test("list-backends returns model readiness metadata for each backend", () => {
		const result = spawnSync("python3", [SCRIPT_PATH, "--list-backends"], {
			encoding: "utf8",
			stdio: ["pipe", "pipe", "pipe"],
			timeout: 15000,
		});

		expect(result.status).toBe(0);
		const backends = JSON.parse(result.stdout);
		expect(Array.isArray(backends)).toBe(true);
		expect(backends.length).toBeGreaterThan(0);

		for (const backend of backends) {
			expect(Array.isArray(backend.models)).toBe(true);
			expect(Array.isArray(backend.installed_models)).toBe(true);
			expect(typeof backend.install_detection).toBe("string");
		}
	});
});
