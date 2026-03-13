/**
 * Voice text processing — command detection & shortcut expansion.
 * Extracted as a pure module for testability.
 */

// ─── Voice Command Detection ─────────────────────────────────────────────

// Longer prefixes first to prevent "hey pi" matching before "hey pie"
const VOICE_COMMAND_PREFIXES = ["hey pie", "hey pi", "pi ", "run ", "execute ", "commit ", "search for ", "open ", "go to "];

const VOICE_COMMAND_MAP: Record<string, (args: string) => string> = {
	"run tests": () => "bun run test",
	"run test": () => "bun run test",
	"run the tests": () => "bun run test",
	"run all tests": () => "bun run test",
	"run typecheck": () => "bun run typecheck",
	"type check": () => "bun run typecheck",
	"run lint": () => "bun run lint",
	"lint this": () => "bun run lint",
	"commit this": () => "git add -A && git commit",
	"commit": () => "git add -A && git commit",
	"git status": () => "git status",
	"git diff": () => "git diff",
	"undo": () => "__UNDO__",
	"undo that": () => "__UNDO__",
	"clear": () => "__CLEAR__",
	"clear all": () => "__CLEAR__",
	"select all": () => "__SELECT_ALL__",
	"new line": () => "__NEWLINE__",
	"submit": () => "__SUBMIT__",
	"send": () => "__SUBMIT__",
	"send it": () => "__SUBMIT__",
};

export interface VoiceCommandResult {
	isCommand: boolean;
	action?: string;
	args?: string;
}

export function detectVoiceCommand(text: string): VoiceCommandResult {
	const lower = text.toLowerCase().trim();

	// Direct command matches
	for (const [trigger, handler] of Object.entries(VOICE_COMMAND_MAP)) {
		if (lower === trigger || lower.replace(/[.,!?]/g, "") === trigger) {
			return { isCommand: true, action: handler("") };
		}
	}

	// Prefix-based commands: "hey pi, run the tests"
	for (const prefix of VOICE_COMMAND_PREFIXES) {
		if (lower.startsWith(prefix)) {
			const rest = lower.slice(prefix.length).replace(/[.,!?]/g, "").trim();
			for (const [trigger, handler] of Object.entries(VOICE_COMMAND_MAP)) {
				if (rest === trigger || rest.startsWith(trigger)) {
					return { isCommand: true, action: handler(rest.slice(trigger.length).trim()) };
				}
			}
			// "hey pi, search for X" → search
			if (rest.startsWith("search for ") || rest.startsWith("search ")) {
				const query = rest.replace(/^search (for )?/, "").trim();
				return { isCommand: true, action: `__SEARCH__${query}` };
			}
			// Generic: "hey pi, <anything>" → send as user message
			if (prefix === "hey pi" || prefix === "hey pie" || prefix === "pi ") {
				return { isCommand: true, action: `__MESSAGE__${rest}` };
			}
		}
	}

	// Voice shortcuts embedded in dictation
	if (lower === "new line" || lower === "newline") return { isCommand: true, action: "__NEWLINE__" };
	if (lower === "submit" || lower === "send it" || lower === "send") return { isCommand: true, action: "__SUBMIT__" };

	return { isCommand: false };
}

// ─── Voice Shortcuts ─────────────────────────────────────────────────────

export function processVoiceShortcuts(text: string): string {
	return text
		.replace(/\bnew line\b/gi, "\n")
		.replace(/\bnewline\b/gi, "\n")
		.replace(/\bperiod\b/gi, ".")
		.replace(/\bcomma\b/gi, ",")
		.replace(/\bquestion mark\b/gi, "?")
		.replace(/\bexclamation mark\b/gi, "!")
		.replace(/\bcolon\b/gi, ":")
		.replace(/\bsemicolon\b/gi, ";")
		.replace(/\bopen parenthesis\b/gi, "(")
		.replace(/\bclose parenthesis\b/gi, ")")
		.replace(/\bopen bracket\b/gi, "[")
		.replace(/\bclose bracket\b/gi, "]")
		.replace(/\bopen brace\b/gi, "{")
		.replace(/\bclose brace\b/gi, "}")
		.replace(/\bbackslash\b/gi, "\\")
		.replace(/\bforward slash\b/gi, "/");
}
