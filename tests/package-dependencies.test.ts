import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const PI_VERSION = "0.74.0";
const PI_DEV_RANGE = `^${PI_VERSION}`;
const PI_PEER_RANGE = "*";

const DIRECT_PI_PACKAGES = ["@earendil-works/pi-coding-agent", "@earendil-works/pi-tui"] as const;
const LEGACY_DIRECT_PI_PACKAGES = ["@mariozechner/pi-coding-agent", "@mariozechner/pi-tui"] as const;
const RELATED_PI_PACKAGES = ["@earendil-works/pi-agent-core", "@earendil-works/pi-ai"] as const;

function readJson(path: string) {
	return JSON.parse(readFileSync(path, "utf8"));
}

function listFiles(dir: string): string[] {
	const files: string[] = [];
	for (const entry of readdirSync(dir)) {
		const fullPath = join(dir, entry);
		const stat = statSync(fullPath);
		if (stat.isDirectory()) {
			files.push(...listFiles(fullPath));
		} else if (/\.(ts|md|json)$/.test(entry)) {
			files.push(fullPath);
		}
	}
	return files;
}

describe("Pi dependency support", () => {
	test("package.json uses the current Earendil Works Pi packages", () => {
		const pkg = readJson("package.json");

		for (const packageName of DIRECT_PI_PACKAGES) {
			expect(pkg.devDependencies?.[packageName]).toBe(PI_DEV_RANGE);
			expect(pkg.peerDependencies?.[packageName]).toBe(PI_PEER_RANGE);
		}

		for (const packageName of LEGACY_DIRECT_PI_PACKAGES) {
			expect(pkg.devDependencies?.[packageName]).toBeUndefined();
			expect(pkg.peerDependencies?.[packageName]).toBeUndefined();
		}
	});

	test("source imports use the current Earendil Works Pi packages", () => {
		const sourceFiles = ["extensions", "tests"]
			.flatMap(listFiles)
			.filter((file) => file !== "tests/package-dependencies.test.ts");

		for (const file of sourceFiles) {
			const contents = readFileSync(file, "utf8");
			expect(contents).not.toContain("@mariozechner/pi-coding-agent");
			expect(contents).not.toContain("@mariozechner/pi-tui");
		}
	});

	test("package-lock.json resolves direct and related Pi packages consistently", () => {
		const lock = readJson("package-lock.json");
		const root = lock.packages?.[""];

		for (const packageName of DIRECT_PI_PACKAGES) {
			expect(root?.devDependencies?.[packageName]).toBe(PI_DEV_RANGE);
			expect(root?.peerDependencies?.[packageName]).toBe(PI_PEER_RANGE);
			expect(lock.packages?.[`node_modules/${packageName}`]?.version).toBe(PI_VERSION);
		}

		for (const packageName of LEGACY_DIRECT_PI_PACKAGES) {
			expect(root?.devDependencies?.[packageName]).toBeUndefined();
			expect(root?.peerDependencies?.[packageName]).toBeUndefined();
			expect(lock.packages?.[`node_modules/${packageName}`]).toBeUndefined();
		}

		for (const packageName of RELATED_PI_PACKAGES) {
			expect(lock.packages?.[`node_modules/${packageName}`]?.version).toBe(PI_VERSION);
		}
	});

	test("bun.lock resolves direct and related Pi packages consistently", () => {
		const lock = readFileSync("bun.lock", "utf8");

		for (const packageName of DIRECT_PI_PACKAGES) {
			expect(lock).toContain(`"${packageName}": "${PI_DEV_RANGE}"`);
			expect(lock).toContain(`"${packageName}": "${PI_PEER_RANGE}"`);
			expect(lock).toContain(`"${packageName}": ["${packageName}@${PI_VERSION}"`);
		}

		for (const packageName of LEGACY_DIRECT_PI_PACKAGES) {
			expect(lock).not.toContain(`"${packageName}": "`);
			expect(lock).not.toContain(`"${packageName}": ["${packageName}@`);
		}

		for (const packageName of RELATED_PI_PACKAGES) {
			expect(lock).toContain(`"${packageName}": ["${packageName}@${PI_VERSION}"`);
		}
	});
});
