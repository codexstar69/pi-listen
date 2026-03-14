/**
 * Voice text processing — command detection & shortcut expansion.
 * Extracted as a pure module for testability.
 */

// ─── Voice Command Detection ─────────────────────────────────────────────

// Longer prefixes first to prevent "hey pi" matching before "hey pie"
const VOICE_COMMAND_PREFIXES = ["hey pie", "hey pi", "pi ", "run ", "execute ", "commit ", "search for ", "open ", "go to ", "switch to ", "change to ", "set "];

const VOICE_COMMAND_MAP: Record<string, (args: string) => string> = {
	// ── Dev commands ──
	"run tests": () => "bun run test",
	"run test": () => "bun run test",
	"run the tests": () => "bun run test",
	"run all tests": () => "bun run test",
	"run typecheck": () => "bun run typecheck",
	"type check": () => "bun run typecheck",
	"run lint": () => "bun run lint",
	"lint this": () => "bun run lint",
	"run build": () => "bun run build",
	"build this": () => "bun run build",
	"build": () => "bun run build",
	"install": () => "bun install",
	"install packages": () => "bun install",
	"install dependencies": () => "bun install",
	"format": () => "bun run format",
	"format this": () => "bun run format",

	// ── Git commands ──
	"commit this": () => "git add -A && git commit",
	"commit": () => "git add -A && git commit",
	"git status": () => "git status",
	"git diff": () => "git diff",
	"git push": () => "git push",
	"push": () => "git push",
	"push this": () => "git push",
	"git pull": () => "git pull",
	"pull": () => "git pull",
	"git log": () => "git log --oneline -20",
	"show log": () => "git log --oneline -20",

	// ── Editor actions ──
	"undo": () => "__UNDO__",
	"undo that": () => "__UNDO__",
	"clear": () => "__CLEAR__",
	"clear all": () => "__CLEAR__",
	"select all": () => "__SELECT_ALL__",
	"new line": () => "__NEWLINE__",
	"submit": () => "__SUBMIT__",
	"send": () => "__SUBMIT__",
	"send it": () => "__SUBMIT__",
	"stop": () => "__STOP__",
	"cancel": () => "__STOP__",
	"abort": () => "__STOP__",

	// ── Pi session commands ──
	"new session": () => "__SLASH__new",
	"start over": () => "__SLASH__new",
	"fresh session": () => "__SLASH__new",
	"new chat": () => "__SLASH__new",
	"compact": () => "__SLASH__compact",
	"compact this": () => "__SLASH__compact",
	"compress": () => "__SLASH__compact",
	"compress context": () => "__SLASH__compact",
	"fork": () => "__SLASH__fork",
	"fork this": () => "__SLASH__fork",
	"fork session": () => "__SLASH__fork",
	"resume": () => "__SLASH__resume",
	"resume session": () => "__SLASH__resume",
	"show tree": () => "__SLASH__tree",
	"session tree": () => "__SLASH__tree",
	"tree": () => "__SLASH__tree",
	"reload": () => "__SLASH__reload",
	"reload extensions": () => "__SLASH__reload",

	// ── Model & thinking ──
	"switch model": () => "__KEY__selectModel",
	"change model": () => "__KEY__selectModel",
	"select model": () => "__KEY__selectModel",
	"pick model": () => "__KEY__selectModel",
	"next model": () => "__KEY__cycleModelForward",
	"previous model": () => "__KEY__cycleModelBackward",
	"cycle model": () => "__KEY__cycleModelForward",
	"more thinking": () => "__KEY__cycleThinkingLevel",
	"cycle thinking": () => "__KEY__cycleThinkingLevel",
	"change thinking": () => "__KEY__cycleThinkingLevel",
	"thinking level": () => "__KEY__cycleThinkingLevel",

	// ── Display toggles ──
	"show thinking": () => "__KEY__toggleThinking",
	"hide thinking": () => "__KEY__toggleThinking",
	"toggle thinking": () => "__KEY__toggleThinking",
	"expand tools": () => "__KEY__expandTools",
	"collapse tools": () => "__KEY__expandTools",
	"show tools": () => "__KEY__expandTools",
	"hide tools": () => "__KEY__expandTools",

	// ── Settings shortcuts ──
	"open settings": () => "__SLASH__settings",
	"settings": () => "__SLASH__settings",
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

			// "switch to <model name>" → model switch
			if (prefix === "switch to " || prefix === "change to ") {
				return { isCommand: true, action: `__MODEL__${rest}` };
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
		.replace(/\bforward slash\b/gi, "/")
		.replace(/\bhash\b/gi, "#")
		.replace(/\bat sign\b/gi, "@")
		.replace(/\bdollar sign\b/gi, "$")
		.replace(/\bampersand\b/gi, "&")
		.replace(/\bpercent\b/gi, "%")
		.replace(/\basterisk\b/gi, "*")
		.replace(/\btab\b/gi, "\t");
}
