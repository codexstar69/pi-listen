import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";

function runPython(script: string) {
	return spawnSync("python3", ["-c", script], {
		cwd: process.cwd(),
		encoding: "utf8",
		stdio: ["pipe", "pipe", "pipe"],
		timeout: 15000,
	});
}

describe("auto backend resolution", () => {
	test("transcribe resolves auto backend to a compatible default model", () => {
		const result = runPython(`
import json
import transcribe
transcribe.detect_backend = lambda: "deepgram"
backend, model = transcribe.resolve_backend_and_model("auto", "small")
print(json.dumps({"backend": backend, "model": model}))
`);

		expect(result.status).toBe(0);
		expect(result.stderr).toBe("");
		expect(JSON.parse(result.stdout)).toEqual({
			backend: "deepgram",
			model: "nova-3",
		});
	});

	test("daemon load normalizes auto backend before caching", () => {
		const result = runPython(`
import json
import transcribe
transcribe.detect_backend = lambda: "deepgram"
import daemon

server = daemon.DaemonServer("/tmp/pi-voice-auto-resolution-test.sock")
server.cache.load = lambda backend, model: {"status": "loaded", "backend": backend, "model": model}
print(json.dumps(server.dispatch({"cmd": "load", "backend": "auto", "model": "small"})))
`);

		expect(result.status).toBe(0);
		expect(result.stderr).toBe("");
		expect(JSON.parse(result.stdout)).toEqual({
			status: "loaded",
			backend: "deepgram",
			model: "nova-3",
		});
	});
});
