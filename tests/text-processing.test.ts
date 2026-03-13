import { describe, expect, test } from "bun:test";
import { detectVoiceCommand, processVoiceShortcuts } from "../extensions/voice/text-processing";

describe("detectVoiceCommand", () => {
	test("detects direct command matches", () => {
		expect(detectVoiceCommand("run tests")).toEqual({ isCommand: true, action: "bun run test" });
		expect(detectVoiceCommand("Run Tests")).toEqual({ isCommand: true, action: "bun run test" });
		expect(detectVoiceCommand("commit")).toEqual({ isCommand: true, action: "git add -A && git commit" });
		expect(detectVoiceCommand("undo")).toEqual({ isCommand: true, action: "__UNDO__" });
		expect(detectVoiceCommand("clear")).toEqual({ isCommand: true, action: "__CLEAR__" });
		expect(detectVoiceCommand("submit")).toEqual({ isCommand: true, action: "__SUBMIT__" });
	});

	test("strips trailing punctuation from direct commands", () => {
		expect(detectVoiceCommand("run tests.")).toEqual({ isCommand: true, action: "bun run test" });
		expect(detectVoiceCommand("commit!")).toEqual({ isCommand: true, action: "git add -A && git commit" });
		expect(detectVoiceCommand("undo?")).toEqual({ isCommand: true, action: "__UNDO__" });
	});

	test("detects prefixed commands", () => {
		expect(detectVoiceCommand("hey pi, run tests")).toEqual({ isCommand: true, action: "bun run test" });
		expect(detectVoiceCommand("hey pi run the tests")).toEqual({ isCommand: true, action: "bun run test" });
		expect(detectVoiceCommand("Hey Pi, Run Lint")).toEqual({ isCommand: true, action: "bun run lint" });
	});

	test("detects search commands", () => {
		const result = detectVoiceCommand("hey pi, search for authentication");
		expect(result.isCommand).toBe(true);
		expect(result.action).toBe("__SEARCH__authentication");
	});

	test("sends generic hey-pi messages as user messages", () => {
		const result = detectVoiceCommand("hey pi, explain this code");
		expect(result.isCommand).toBe(true);
		expect(result.action).toBe("__MESSAGE__explain this code");
	});

	test("returns not-a-command for regular speech", () => {
		expect(detectVoiceCommand("I need to fix the login page")).toEqual({ isCommand: false });
		expect(detectVoiceCommand("the test results look good")).toEqual({ isCommand: false });
		expect(detectVoiceCommand("")).toEqual({ isCommand: false });
	});

	test("detects standalone shortcut commands", () => {
		expect(detectVoiceCommand("new line")).toEqual({ isCommand: true, action: "__NEWLINE__" });
		expect(detectVoiceCommand("newline")).toEqual({ isCommand: true, action: "__NEWLINE__" });
		expect(detectVoiceCommand("send it")).toEqual({ isCommand: true, action: "__SUBMIT__" });
		expect(detectVoiceCommand("send")).toEqual({ isCommand: true, action: "__SUBMIT__" });
	});

	test("is case-insensitive", () => {
		expect(detectVoiceCommand("RUN TESTS")).toEqual({ isCommand: true, action: "bun run test" });
		expect(detectVoiceCommand("Hey Pi, Git Status")).toEqual({ isCommand: true, action: "git status" });
	});

	test("handles hey pie (common misrecognition)", () => {
		const result = detectVoiceCommand("hey pie, run tests");
		expect(result.isCommand).toBe(true);
		expect(result.action).toBe("bun run test");
	});

	test("all direct command variants", () => {
		expect(detectVoiceCommand("run test").action).toBe("bun run test");
		expect(detectVoiceCommand("run the tests").action).toBe("bun run test");
		expect(detectVoiceCommand("run all tests").action).toBe("bun run test");
		expect(detectVoiceCommand("run typecheck").action).toBe("bun run typecheck");
		expect(detectVoiceCommand("type check").action).toBe("bun run typecheck");
		expect(detectVoiceCommand("run lint").action).toBe("bun run lint");
		expect(detectVoiceCommand("lint this").action).toBe("bun run lint");
		expect(detectVoiceCommand("commit this").action).toBe("git add -A && git commit");
		expect(detectVoiceCommand("git status").action).toBe("git status");
		expect(detectVoiceCommand("git diff").action).toBe("git diff");
		expect(detectVoiceCommand("undo that").action).toBe("__UNDO__");
		expect(detectVoiceCommand("clear all").action).toBe("__CLEAR__");
		expect(detectVoiceCommand("select all").action).toBe("__SELECT_ALL__");
	});
});

describe("processVoiceShortcuts", () => {
	// Note: processVoiceShortcuts replaces the word only, not surrounding spaces.
	// "hello period" → "hello ." — the space before the word remains.
	// This matches how voice transcription works (words are space-separated).

	test("replaces punctuation shortcut words", () => {
		expect(processVoiceShortcuts("hello period")).toBe("hello .");
		expect(processVoiceShortcuts("what question mark")).toBe("what ?");
		expect(processVoiceShortcuts("wow exclamation mark")).toBe("wow !");
		expect(processVoiceShortcuts("one comma two")).toBe("one , two");
		expect(processVoiceShortcuts("note colon important")).toBe("note : important");
		expect(processVoiceShortcuts("first semicolon second")).toBe("first ; second");
	});

	test("replaces bracket shortcut words", () => {
		expect(processVoiceShortcuts("open parenthesis x close parenthesis")).toBe("( x )");
		expect(processVoiceShortcuts("open bracket 0 close bracket")).toBe("[ 0 ]");
		expect(processVoiceShortcuts("open brace close brace")).toBe("{ }");
	});

	test("replaces newline shortcut words", () => {
		expect(processVoiceShortcuts("line one new line line two")).toBe("line one \n line two");
		expect(processVoiceShortcuts("first newline second")).toBe("first \n second");
	});

	test("replaces slash shortcut words", () => {
		expect(processVoiceShortcuts("path forward slash file")).toBe("path / file");
		expect(processVoiceShortcuts("escape backslash n")).toBe("escape \\ n");
	});

	test("handles multiple shortcuts in one string", () => {
		const input = "function open parenthesis x comma y close parenthesis";
		expect(processVoiceShortcuts(input)).toBe("function ( x , y )");
	});

	test("is case-insensitive", () => {
		expect(processVoiceShortcuts("Hello Period")).toBe("Hello .");
		expect(processVoiceShortcuts("test NEW LINE done")).toBe("test \n done");
	});

	test("leaves unrecognized text unchanged", () => {
		expect(processVoiceShortcuts("normal text here")).toBe("normal text here");
		expect(processVoiceShortcuts("")).toBe("");
	});

	test("handles words at start and end of string", () => {
		expect(processVoiceShortcuts("period at the start")).toBe(". at the start");
		expect(processVoiceShortcuts("end with period")).toBe("end with .");
	});
});
