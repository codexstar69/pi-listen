/**
 * pi-lumo — Lumo Companion Extension for Pi CLI.
 *
 * A 3D raymarched virtual pet that lives above the editor.
 * Interactive commands: pet, feed, ball, music, color, sleep, wake, flip, hide.
 * Reacts to voice recording audio levels when pi-voice is active.
 *
 * Install alongside @codexstar/pi-listen for voice-reactive animations,
 * or use standalone as a terminal companion.
 */

import type {
	ExtensionAPI,
	ExtensionContext,
} from "@mariozechner/pi-coding-agent";

import { renderLumo, resetLumo, lumoSetTalking, lumoKeypress } from "./voice/lumo";

export default function (pi: ExtensionAPI) {
	let ctx: ExtensionContext | null = null;
	let companionTimer: ReturnType<typeof setInterval> | null = null;
	let companionActive = false;
	let lastRenderTime = Date.now();
	let terminalInputUnsub: (() => void) | null = null;
	let enabled = true;

	function showCompanion() {
		if (companionActive) return;
		if (!ctx?.hasUI) return;
		companionActive = true;
		lastRenderTime = Date.now();

		const renderCompanion = (width: number): string[] => {
			const now = Date.now();
			const dt = Math.min(0.1, (now - lastRenderTime) / 1000);
			lastRenderTime = now;
			return renderLumo(Math.min(width, 60), 0, dt);
		};

		ctx.ui.setWidget("lumo-companion", (_tui, _theme) => ({
			invalidate() {},
			render: renderCompanion,
		}), { placement: "aboveEditor" });

		companionTimer = setInterval(() => {
			if (!ctx?.hasUI) return;
			ctx.ui.setWidget("lumo-companion", (_tui, _theme) => ({
				invalidate() {},
				render: renderCompanion,
			}), { placement: "aboveEditor" });
		}, 150);
	}

	function hideCompanion() {
		companionActive = false;
		if (companionTimer) {
			clearInterval(companionTimer);
			companionTimer = null;
		}
		lumoSetTalking(false);
		if (ctx?.hasUI) {
			ctx.ui.setWidget("lumo-companion", undefined);
		}
	}

	function setupKeyHandler() {
		if (!ctx?.hasUI) return;
		if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }

		terminalInputUnsub = ctx.ui.onTerminalInput((data: string) => {
			if (!enabled || !companionActive) return undefined;

			// Forward single-char commands to Lumo when editor is empty
			if (data.length === 1 && "pfbmcdswo".includes(data)) {
				const editorText = ctx?.hasUI ? (ctx.ui.getEditorText() || "") : "";
				if (!editorText.trim()) {
					lumoKeypress(data);
					return { consume: true };
				}
			}

			return undefined;
		});
	}

	// ─── Lifecycle ───────────────────────────────────────────────────────────

	pi.on("session_start", async (_event, startCtx) => {
		ctx = startCtx;
		if (enabled) {
			showCompanion();
			setupKeyHandler();
		}
	});

	pi.on("session_shutdown", async () => {
		hideCompanion();
		resetLumo();
		if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }
	});

	pi.on("session_switch", async (_event, switchCtx) => {
		hideCompanion();
		resetLumo();
		ctx = switchCtx;
		if (enabled) {
			showCompanion();
			setupKeyHandler();
		}
	});

	// ─── /lumo command ───────────────────────────────────────────────────────

	const lumoCommands: Record<string, string> = {
		pet: "p", feed: "f", ball: "b", music: "m", color: "c",
		sleep: "s", wake: "w", flip: "d", hide: "o",
	};

	pi.registerCommand("lumo", {
		description: "Lumo companion: /lumo [on|off|pet|feed|ball|music|color|sleep|wake|flip|hide]",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const sub = (args || "").trim().toLowerCase();

			if (sub === "on") {
				enabled = true;
				showCompanion();
				setupKeyHandler();
				cmdCtx.ui.notify("Lumo companion enabled. 🐾", "info");
				return;
			}

			if (sub === "off") {
				enabled = false;
				hideCompanion();
				resetLumo();
				if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }
				cmdCtx.ui.notify("Lumo companion hidden.", "info");
				return;
			}

			if (lumoCommands[sub]) {
				if (!companionActive) {
					enabled = true;
					showCompanion();
					setupKeyHandler();
				}
				lumoKeypress(lumoCommands[sub]);
				return;
			}

			// Default: toggle
			if (companionActive) {
				enabled = false;
				hideCompanion();
				resetLumo();
				cmdCtx.ui.notify("Lumo companion hidden.", "info");
			} else {
				enabled = true;
				showCompanion();
				setupKeyHandler();
				cmdCtx.ui.notify("Lumo companion enabled. 🐾", "info");
			}
		},
	});
}
