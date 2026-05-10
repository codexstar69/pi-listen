import { describe, expect, test } from "bun:test";
import { DEFAULT_CONFIG } from "../extensions/voice/config";
import { buildDeepgramWsUrl } from "../extensions/voice/deepgram";

describe("buildDeepgramWsUrl", () => {
	test("adds Deepgram keyterms from config", () => {
		const url = new URL(buildDeepgramWsUrl({
			...DEFAULT_CONFIG,
			language: "pt-BR",
			deepgramKeyterms: ["Raycast", "Linear", "VS Code"],
		}));

		expect(url.searchParams.getAll("keyterm")).toEqual(["Raycast", "Linear", "VS Code"]);
	});

	test("skips blank Deepgram keyterms", () => {
		const url = new URL(buildDeepgramWsUrl({
			...DEFAULT_CONFIG,
			deepgramKeyterms: ["", "  ", "Cursor"],
		}));

		expect(url.searchParams.getAll("keyterm")).toEqual(["Cursor"]);
	});
});
