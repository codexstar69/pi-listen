/**
 * Voice Settings Panel — interactive overlay for backend / model / language
 * configuration. Tab navigation across General / Models / Downloaded / Device.
 *
 * Architecture:
 *   - Component interface: render(width) / handleInput(data) / invalidate()
 *   - Opened via ctx.ui.custom() with overlay: true
 *   - Tab switch with ←→, row navigation with ↑↓
 *   - Inline language sub-picker with fuzzy search
 *   - Theme-aware colors via the host Theme passed in PanelDeps; falls back
 *     to raw ANSI when theme is not provided so unit tests / mock harnesses
 *     don't have to construct one
 *   - Delete on Downloaded tab is two-step (`x` once arms, `x` within 1.5s
 *     confirms) so a stray keypress can't nuke a multi-GB download
 *   - Render cache deliberately omitted — the panel renders ~12-30 lines and
 *     mutating any of: tab, row, search, sub-picker state, delete confirm
 *     timer, active model would invalidate it; the cost of always-rendering
 *     is far below the cost of stale frames
 */

import { matchesKey, Key, truncateToWidth } from "@earendil-works/pi-tui";
import type { Theme, ThemeColor } from "@earendil-works/pi-coding-agent";
import type { VoiceConfig, VoiceSettingsScope } from "./config";
import { LOCAL_MODELS, getLanguagesForLocalModel, type LocalModelInfo } from "./local";
import type { DeviceProfile, ModelFitness } from "./device";
import { getFreeDiskSpace, formatBytes, getModelsDir, scanHandyModels, importHandyModel } from "./model-download";
import {
	TTS_LOCAL_MODELS as TTS_LOCAL_MODELS_REF,
	isTtsModelInstalled as TTS_INSTALLED_CHECK_REF,
	type TtsLocalModelInfo,
	type TtsVoice,
} from "./tts-local-models";
import {
	DEEPGRAM_TTS_VOICES,
	filterDeepgramVoicesByLanguage,
} from "./tts-deepgram";
import { PickerChassis, type PickerRow } from "./ui-picker";
import { ICON } from "./ui-icons";
import { localeLabel, formatRomanizedLabel } from "./ui-locale-labels";
import { visualWidth, isPanelTooNarrow, widthTier } from "./ui-width";

/**
 * v7.1 — Build PickerChassis rows for the TTS Models picker, grouping
 * the catalog into Recommended / Per-language / Multilingual sections.
 * Rows are produced in a stable order; headings appear only when at
 * least one model in the group is present (handled by the chassis at
 * filter time, but we also skip empty groups upfront).
 */
function buildTtsModelPickerRows(catalog: ReadonlyArray<TtsLocalModelInfo>): PickerRow<TtsLocalModelInfo>[] {
	const recommended = catalog.filter(m => m.preferred === true);
	const perLanguage = catalog.filter(m => !m.preferred && m.tier === "edge");
	const heavy = catalog.filter(m => m.tier !== "edge");
	const out: PickerRow<TtsLocalModelInfo>[] = [];
	const pushGroup = (heading: string, rows: TtsLocalModelInfo[]) => {
		if (rows.length === 0) return;
		out.push({ kind: "heading", label: heading });
		for (const m of rows) {
			const sk = `${m.name} ${m.id} ${m.notes} ${m.languages.join(" ")}`;
			out.push({ kind: "data", value: m, searchKey: sk });
		}
	};
	pushGroup("Recommended", recommended);
	pushGroup("Per-language voices", perLanguage);
	pushGroup("Multilingual / heavyweight", heavy);
	return out;
}

// ─── Types ────────────────────────────────────────────────────────────────────

const TAB_IDS = ["general", "models", "downloaded", "speak", "device"] as const;
const TAB_LABELS = ["General", "Models", "Downloaded", "Speak", "Device"];
type TabId = (typeof TAB_IDS)[number];

export type PanelAction =
	| { type: "download"; modelId: string }
	| { type: "speak-test" }
	| { type: "tts-install"; modelId: string }
	| undefined;

export interface PanelDeps {
	config: VoiceConfig;
	device: DeviceProfile;
	cwd: string;
	getModelFitness: (m: LocalModelInfo, d: DeviceProfile) => ModelFitness;
	getDownloadedModels: () => { id: string; sizeMB: number }[];
	deleteModel: (id: string) => boolean;
	isSherpaAvailable: () => boolean;
	formatDeviceSummary: (d: DeviceProfile) => string;
	saveConfig: (config: VoiceConfig, scope: VoiceSettingsScope, cwd: string) => void;
	clearRecognizerCache: () => void;
	resolveApiKey: () => string | undefined;
	deepgramLanguages: { name: string; code: string; popular?: boolean }[];
	/**
	 * Optional. If provided, panel renders use the host theme so colors track
	 * user theme choices (Catppuccin, Solarized, etc.). Without it, raw ANSI
	 * approximations are used — fine for tests, slightly less polished in
	 * a real Pi session.
	 */
	theme?: Theme;
}

interface ModelRow extends LocalModelInfo {
	fitness: ModelFitness;
}

/**
 * Visual grouping of LOCAL_MODELS in the Models tab. Each group has a
 * heading and the list under it preserves catalog order for stable display.
 * "Top picks" is computed dynamically from device fitness — see groupModels().
 */
interface ModelGroup {
	heading: string;
	subtitle?: string;
	rows: ModelRow[];
}

// Section headings drive both the visual divider and the row indexing for
// `↑↓` navigation. Empty groups are dropped so the user never sees a stray
// header with no entries (e.g. "Top picks" when nothing fits the device).

// ─── Panel ────────────────────────────────────────────────────────────────────

export class VoiceSettingsPanel {
	onClose?: (result?: PanelAction) => void;

	private tab = 0;
	private row = 0;
	private sub: "main" | "lang-picker" | "tts-model-picker" | "tts-voice-picker" = "main";

	// Models tab — grouped view
	private modelSearch = "";
	private modelGroups: ModelGroup[] = [];
	/** Flat row list synthesized from `modelGroups`; navigation skips headings. */
	private modelRows: { type: "heading"; group: ModelGroup } | ModelRow[] = [] as never;
	private modelRowsFlat: ({ kind: "heading"; group: ModelGroup } | { kind: "row"; row: ModelRow })[] = [];
	/** Selectable indexes (i.e. row entries only — headings excluded). */
	private modelSelectableIdx: number[] = [];

	// Language sub-picker
	private langSearch = "";
	private langList: { name: string; code: string }[] = [];
	private langFiltered: { name: string; code: string }[] = [];
	private langRow = 0;

	// TTS model sub-picker (Speak tab → Model row).
	// v7.1: backed by PickerChassis with v7.1 grouping (§3 + §7).
	private ttsModelChassis: PickerChassis<TtsLocalModelInfo> | null = null;

	// TTS voice sub-picker (Speak tab → Voice row)
	private ttsVoiceSearch = "";
	private ttsVoiceRow = 0;

	// Two-step delete on the Downloaded tab. When `x` is pressed, set the
	// pending modelId + expiry timestamp; a second `x` within DELETE_CONFIRM_MS
	// commits. Any other navigation cancels.
	private deletePendingId: string | null = null;
	private deletePendingExpiresAt = 0;
	private static readonly DELETE_CONFIRM_MS = 1500;

	constructor(private p: PanelDeps, initialTab?: number) {
		if (initialTab !== undefined && initialTab >= 0 && initialTab < TAB_IDS.length) {
			this.tab = initialTab;
		}
		this.rebuildModels();
	}

	// ─── Theme-aware color helpers ────────────────────────────────────────
	//
	// Each helper prefers theme.fg() when a Theme was passed via PanelDeps;
	// otherwise it emits raw ANSI as a fallback so tests / non-interactive
	// surfaces still get visually distinct text.

	private c(role: ThemeColor, ansiFallback: string, s: string): string {
		const theme = this.p.theme;
		if (theme) return theme.fg(role, s);
		return `\x1b[${ansiFallback}m${s}\x1b[39m`;
	}
	private dim = (s: string) => this.p.theme ? this.p.theme.fg("dim", s) : `\x1b[2m${s}\x1b[22m`;
	private bold = (s: string) => `\x1b[1m${s}\x1b[22m`;
	private accent = (s: string) => this.c("accent", "36", s);
	private success = (s: string) => this.c("success", "32", s);
	private warning = (s: string) => this.c("warning", "33", s);
	private error = (s: string) => this.c("error", "31", s);
	private muted = (s: string) => this.p.theme ? this.p.theme.fg("muted", s) : `\x1b[2m${s}\x1b[22m`;

	// ─── Component interface ──────────────────────────────────────────────

	render(width: number): string[] {
		// v7.1 §10/§13 — hard block below 60 cols. The panel cannot
		// degrade below tier 3 cleanly without per-component reflow
		// logic that doubles picker chassis complexity. Below tier 3
		// the user sees a single-line "resize" message and slash
		// commands remain available (panel-less).
		if (isPanelTooNarrow(width)) {
			return [
				`  ${this.bold("pi-listen")}`,
				`  ${this.warning(`Terminal too narrow ${ICON.middot} resize to ${ICON.arrowRight} 60 cols`)}`,
				`  ${this.dim("Slash commands still work: /voice-speak, /voice-models, /voice-settings")}`,
			];
		}

		const w = Math.max(36, Math.min(width - 2, 80));
		const iw = w - 4;
		const t = (s: string) => truncateToWidth(s, w);

		const lines: string[] = [];
		const { device } = this.p;

		// v7.1 §12 — two-row status header.
		// Row 1: brand + device summary (existing).
		// Row 2: at-a-glance system state — STT backend/model + TTS
		// backend/model/voice + active language. Width-tier aware:
		// at "mid" (60-79) the row is trimmed; at "wide" (≥80) it
		// shows the full picture.
		lines.push(t(`  ${this.bold("pi-listen")}  ${this.dim(this.p.formatDeviceSummary(device))}`));
		lines.push(t("  " + this.renderStatusRow(widthTier(width) === "mid")));
		lines.push(t(this.dim("  " + "─".repeat(Math.min(iw, 60)))));

		// Tab bar — underline-style active indicator (no brackets noise)
		lines.push(t("  " + this.renderTabBar()));
		lines.push("");

		// Sub-mode: language picker takes over the body
		if (this.sub === "lang-picker") {
			lines.push(...this.renderLangPicker(w, iw).map(t));
			return lines;
		}
		if (this.sub === "tts-model-picker") {
			lines.push(...this.renderTtsModelPicker(w, iw).map(t));
			return lines;
		}
		if (this.sub === "tts-voice-picker") {
			lines.push(...this.renderTtsVoicePicker(w, iw).map(t));
			return lines;
		}

		// Tab content
		const tabId = TAB_IDS[this.tab]!;
		switch (tabId) {
			case "general":
				lines.push(...this.renderGeneral(w, iw).map(t));
				break;
			case "models":
				lines.push(...this.renderModels(w, iw).map(t));
				break;
			case "downloaded":
				lines.push(...this.renderDownloaded(w, iw).map(t));
				break;
			case "speak":
				lines.push(...this.renderSpeak(w, iw).map(t));
				break;
			case "device":
				lines.push(...this.renderDevice(w, iw).map(t));
				break;
		}

		return lines;
	}

	handleInput(data: string): void {
		if (this.sub === "lang-picker") {
			this.handleLangInput(data);
			return;
		}
		if (this.sub === "tts-model-picker") {
			this.handleTtsModelInput(data);
			return;
		}
		if (this.sub === "tts-voice-picker") {
			this.handleTtsVoiceInput(data);
			return;
		}

		const tabId = TAB_IDS[this.tab]!;

		// Any key other than `x` while a delete is pending cancels the confirmation
		// (so the user can navigate around without committing destructive action).
		if (this.deletePendingId && data !== "x") {
			this.deletePendingId = null;
			this.deletePendingExpiresAt = 0;
		}

		// Tab navigation: ←→ or Tab
		if (matchesKey(data, Key.left)) {
			this.tab = (this.tab - 1 + TAB_IDS.length) % TAB_IDS.length;
			this.row = 0;
			this.modelSearch = "";
			this.refreshModelView();
			return;
		}
		if (matchesKey(data, Key.right) || data === "\t") {
			this.tab = (this.tab + 1) % TAB_IDS.length;
			this.row = 0;
			this.modelSearch = "";
			this.refreshModelView();
			return;
		}

		// Close
		if (matchesKey(data, Key.escape)) {
			this.onClose?.();
			return;
		}

		// Row navigation: ↑↓
		if (matchesKey(data, Key.up)) {
			const max = this.getRowCount(tabId);
			if (max > 0) this.row = this.row === 0 ? max - 1 : this.row - 1;
			return;
		}
		if (matchesKey(data, Key.down)) {
			const max = this.getRowCount(tabId);
			if (max > 0) this.row = this.row === max - 1 ? 0 : this.row + 1;
			return;
		}

		// Enter = select/toggle
		if (matchesKey(data, Key.enter)) {
			this.handleSelect(tabId);
			return;
		}

		// Tab-specific keys
		if (tabId === "models") {
			if (matchesKey(data, Key.backspace)) {
				this.modelSearch = this.modelSearch.slice(0, -1);
				this.refreshModelView();
				this.row = 0;
				return;
			}
			if (data.length === 1 && data >= " " && data <= "~") {
				this.modelSearch += data;
				this.refreshModelView();
				this.row = 0;
				return;
			}
		}

		if (tabId === "downloaded") {
			if (data === "x" || data === "d") {
				this.handleDeleteRequest();
				return;
			}
		}
	}

	/**
	 * Kept for backward-compat with the host TUI which may still call
	 * invalidate() to force a re-render. Renders are cheap and have no cache,
	 * so this is a no-op.
	 */
	invalidate(): void { /* render is uncached */ }

	// ─── v7.1 §12 — Two-row status header ─────────────────────────────────

	/**
	 * Render the at-a-glance second header row showing current system
	 * state. Format depends on width tier:
	 *   wide (≥80): STT <on/off> · <stt-backend>/<stt-model> · TTS <on/off> · <tts-backend>/<tts-model>/<voice> · Lang <lang>
	 *   mid (60-79): STT <on/off> · <stt-backend> · TTS <on/off> · <tts-backend> · <lang>
	 * Each segment is dim-by-default, on/off badge tinted by state.
	 */
	private renderStatusRow(compact: boolean): string {
		const { config } = this.p;
		const sttOn = config.enabled === true;
		const ttsOn = config.ttsEnabled === true;
		const sttBackend = sttOn ? (config.backend === "deepgram" ? "Deepgram" : "Local") : "off";
		const ttsBackend = ttsOn ? ((config.ttsBackend ?? "local") === "deepgram" ? "Deepgram" : "Local") : "off";
		const lang = config.ttsLanguage || config.language || "en";
		const sttBadge = sttOn ? this.success(`${ICON.bulletActive} STT`) : this.dim(`${ICON.bulletInactive} STT`);
		const ttsBadge = ttsOn ? this.success(`${ICON.bulletActive} TTS`) : this.dim(`${ICON.bulletInactive} TTS`);
		const sep = this.dim(`  ${ICON.middot}  `);
		if (compact) {
			return `${sttBadge} ${this.dim(sttBackend)}${sep}${ttsBadge} ${this.dim(ttsBackend)}${sep}${this.dim(`Lang ${lang}`)}`;
		}
		// wide tier — include model/voice details
		const sttDetail = sttOn
			? `${sttBackend}${config.backend === "local" && config.localModel ? `/${this.shortenId(config.localModel)}` : ""}`
			: "off";
		const ttsModelId = ttsOn && (config.ttsBackend ?? "local") === "local"
			? this.shortenId(config.ttsLocalModel ?? "")
			: ttsOn && (config.ttsBackend ?? "local") === "deepgram"
				? this.shortenId(String(config.ttsDeepgramVoiceId ?? ""))
				: "";
		const ttsDetail = ttsOn ? `${ttsBackend}${ttsModelId ? `/${ttsModelId}` : ""}` : "off";
		return `${sttBadge} ${this.dim(sttDetail)}${sep}${ttsBadge} ${this.dim(ttsDetail)}${sep}${this.dim(`Lang ${lang}`)}`;
	}

	private shortenId(id: string): string {
		// Strip common prefixes for readable status row entries.
		return id.replace(/^vits-piper-/, "piper-").replace(/-int8|-q8/, "").slice(0, 22);
	}

	// ─── Tab bar ──────────────────────────────────────────────────────────

	private renderTabBar(): string {
		// Inactive tabs are dim, active tab is accent + bold, with an underline
		// rule below to anchor the eye. Equal-width spacing makes ←→ feel
		// consistent even when tab labels differ in length.
		return TAB_LABELS.map((label, i) => {
			if (i === this.tab) return this.accent(this.bold(label));
			return this.dim(label);
		}).join(this.dim("  ·  "));
	}

	// ─── General tab ──────────────────────────────────────────────────────

	private renderGeneral(_w: number, iw: number): string[] {
		const lines: string[] = [];
		const { config } = this.p;
		const isLocal = config.backend === "local";
		const useShort = iw < 42;

		const rows: { label: string; value: string; hint?: string }[] = [
			{
				label: "Backend",
				value: isLocal
					? this.success("Local (offline, batch)")
					: this.accent("Deepgram (cloud, live streaming)"),
				hint: "toggle",
			},
			{
				label: isLocal ? "Model" : "API Key",
				value: isLocal
					? (LOCAL_MODELS.find(m => m.id === config.localModel)?.name || config.localModel || "—")
					: (() => {
						const key = this.p.resolveApiKey();
						return key ? this.success(`set (${key.slice(0, 8)}…)`) : this.error("NOT SET");
					})(),
				hint: isLocal ? "choose ›" : undefined,
			},
			{
				label: "Language",
				value: this.getLangDisplay(),
				hint: "change",
			},
			{
				label: "Scope",
				value: config.scope === "project"
					? (useShort ? "Project" : "Project (this repo)")
					: (useShort ? "Global" : "Global (all projects)"),
				hint: "toggle",
			},
			{
				label: "Voice",
				value: config.enabled ? this.success("Enabled") : this.error("Disabled"),
				hint: "toggle",
			},
			{
				// v7.1.3: STT auto-submit toggle. When ON, transcribed
				// text is sent directly to the agent (turn triggered)
				// instead of being placed in the editor for the user
				// to press [enter].
				label: "Auto-send",
				value: config.autoSubmitOnSpeak === true
					? this.success("ON — STT speaks the message immediately")
					: this.dim("OFF — STT fills the editor (press ↵ to send)"),
				hint: "toggle",
			},
		];

		// v7.2 — left-bar cursor + dim non-selected (HIG deference).
		const labelW = 12;
		for (let i = 0; i < rows.length; i++) {
			const r = rows[i]!;
			const isSelected = i === this.row;
			const prefix = isSelected ? `${this.accent(ICON.cursorBar)}  ` : `   `;
			const label = isSelected ? r.label.padEnd(labelW) : this.dim(r.label.padEnd(labelW));
			const hint = (isSelected && r.hint) ? this.dim(` [↵ ${r.hint}]`) : "";
			lines.push(`${prefix}${label}${r.value}${hint}`);
		}

		lines.push("");
		lines.push(this.dim("  ↵ change  ←→/Tab tabs  ↑↓ navigate  esc close"));
		return lines;
	}

	// ─── Models tab ───────────────────────────────────────────────────────

	private renderModels(_w: number, iw: number): string[] {
		const lines: string[] = [];
		const currentId = this.p.config.localModel || "parakeet-v3";
		const downloadedMap = new Map(this.p.getDownloadedModels().map(d => [d.id, d.sizeMB]));

		// Search bar
		const cursor = this.modelSearch ? this.modelSearch : this.dim("type to search…");
		lines.push(`  ${this.dim("Search:")} ${cursor}`);
		lines.push("");

		// Models are flattened with headings interleaved. We draw a viewport
		// window around the selected row, but always keep the heading of the
		// current group visible above so the user has context.
		if (this.modelRowsFlat.length === 0) {
			lines.push(this.dim("    No matching models"));
			lines.push("");
			lines.push(this.dim("  ↵ activate / download  ←→ tabs  ↑↓ navigate  esc close"));
			return lines;
		}

		const maxVisible = 14;
		const totalSelectable = this.modelSelectableIdx.length;
		const selectedFlatIdx = this.modelSelectableIdx[Math.min(this.row, totalSelectable - 1)] ?? 0;

		// Find a viewport that includes the selected row plus surrounding context.
		// We center on selection but always show the parent heading.
		const halfWindow = Math.floor(maxVisible / 2);
		let start = Math.max(0, selectedFlatIdx - halfWindow);
		let end = Math.min(this.modelRowsFlat.length, start + maxVisible);
		if (end - start < maxVisible) {
			start = Math.max(0, end - maxVisible);
		}

		// Bring the heading immediately above `start` into view if it isn't
		// already, so the user always knows which group they're in.
		while (start > 0 && this.modelRowsFlat[start - 1]?.kind === "heading") start--;

		// Right-align size column. Using a fixed character budget keeps row
		// boundaries clean across model name lengths.
		const nameW = Math.min(28, Math.max(18, iw - 30));

		for (let i = start; i < end; i++) {
			const item = this.modelRowsFlat[i];
			if (!item) continue;
			if (item.kind === "heading") {
				const heading = item.group.heading;
				const subtitle = item.group.subtitle ? this.dim(`  ${item.group.subtitle}`) : "";
				lines.push("");
				lines.push(`  ${this.dim(this.bold(heading.toUpperCase()))}${subtitle}`);
				continue;
			}
			const m = item.row;
			const isSelected = i === selectedFlatIdx;
			const isCurrent = m.id === currentId;
			const isDl = downloadedMap.has(m.id);

			const prefix = isSelected ? `${this.accent(ICON.cursorBar)}  ` : `   `;
			const name = (isSelected ? this.accent(m.name) : m.name).padEnd(nameW + (isSelected ? 0 : 0));
			const namePad = m.name.length < nameW ? " ".repeat(nameW - m.name.length) : "";
			const size = this.dim(m.size.padStart(8));
			const langHint = this.dim(formatLangHint(m).padEnd(13));
			const status = isCurrent
				? this.success("active")
				: isDl
					? this.success("ready")
					: this.dim(formatFitness(m.fitness));

			lines.push(`${prefix}${isSelected ? this.accent(m.name) + namePad : m.name + namePad} ${size}  ${langHint} ${status}`);

			// Expanded detail under the selected row only — accuracy/speed
			// bars + freeform notes. Avoids the previous redundancy of
			// showing compact ratings on every row.
			if (isSelected) {
				const accBar = this.ratingBar(m.accuracy, "accuracy");
				const spdBar = this.ratingBar(m.speed, "speed   ");
				lines.push(`        ${accBar}   ${spdBar}`);
				lines.push(`        ${this.dim(m.notes)}`);
			}
		}

		// Scroll affordance
		if (start > 0 || end < this.modelRowsFlat.length) {
			lines.push(this.dim(`    showing ${start + 1}–${end} of ${this.modelRowsFlat.length}`));
		}

		lines.push("");
		const selectedRow = this.getRowAt(this.row);
		const enterHint = selectedRow
			? (downloadedMap.has(selectedRow.id) ? "activate" : `download (${selectedRow.size}) + activate`)
			: "select";
		lines.push(this.dim(`  ↵ ${enterHint}  ←→/Tab tabs  ↑↓ navigate  esc close`));
		return lines;
	}

	// ─── Downloaded tab ───────────────────────────────────────────────────

	private renderDownloaded(_w: number, _iw: number): string[] {
		const lines: string[] = [];
		const dl = this.getDownloaded();
		const currentId = this.p.config.localModel || "parakeet-v3";
		const handy = scanHandyModels();
		const handyNotImported = handy.filter(h => !h.imported);

		// Auto-expire pending delete confirmation. Re-rendering with an old
		// pending state is a no-op — the renderer just won't show the badge.
		if (this.deletePendingId && Date.now() > this.deletePendingExpiresAt) {
			this.deletePendingId = null;
		}

		if (dl.length === 0 && handyNotImported.length === 0) {
			lines.push(this.dim("    No downloaded models yet."));
			lines.push(this.dim("    Models download automatically on first recording."));
			lines.push(this.dim("    Use the Models tab to browse and install."));
		} else {
			// Pi-managed models
			if (dl.length > 0) {
				let totalMB = 0;
				for (let i = 0; i < dl.length; i++) {
					const d = dl[i]!;
					totalMB += d.sizeMB;
					const isSelected = i === this.row;
					const isCurrent = d.id === currentId;
					const isDeletePending = d.id === this.deletePendingId;
					const prefix = isSelected ? `${this.accent(ICON.cursorBar)}  ` : `   `;
					const name = isSelected ? this.accent(d.name) : d.name;
					const size = this.dim(` — ${d.sizeMB} MB`);
					const status = isCurrent ? this.success(" active") : "";
					const deleteBadge = isDeletePending
						? "  " + this.warning("press x again to delete")
						: "";
					lines.push(`${prefix}${name}${size}${status}${deleteBadge}`);
				}
				lines.push(this.dim(`    Total: ${totalMB} MB on disk`));
			}

			// Handy import section
			if (handyNotImported.length > 0) {
				lines.push("");
				lines.push(this.dim("    Available from Handy"));
				for (let i = 0; i < handyNotImported.length; i++) {
					const h = handyNotImported[i]!;
					const idx = dl.length + i;
					const isSelected = idx === this.row;
					const prefix = isSelected ? `${this.accent(ICON.cursorBar)}  ` : `   `;
					const name = isSelected ? this.accent(h.name) : h.name;
					const size = this.dim(` — ${h.sizeMB} MB`);
					lines.push(`${prefix}${name}${size}${this.warning(" ↵ import")}`);
				}
			}
		}

		lines.push("");
		const hasItems = dl.length > 0 || handyNotImported.length > 0;
		const hint = hasItems
			? "  ↵ activate/import  x delete (press twice)  ←→/Tab tabs  ↑↓ navigate  esc close"
			: "  ←→/Tab tabs  esc close";
		lines.push(this.dim(hint));
		return lines;
	}

	// ─── Speak tab (TTS) ──────────────────────────────────────────────────

	private renderSpeak(_w: number, _iw: number): string[] {
		const lines: string[] = [];
		const { config } = this.p;
		const isLocal = (config.ttsBackend ?? "local") === "local";

		// Always-visible status line — single source of truth for the
		// current TTS configuration so the user can scan it without
		// reading every row.
		const statusParts: string[] = [];
		statusParts.push(isLocal ? "Local" : "Deepgram");
		statusParts.push(this.formatActiveModelOrVoice(config));
		statusParts.push(`${(config.ttsSpeed ?? 1.0).toFixed(2)}×`);
		const lang = config.ttsLanguage || config.language || "en";
		statusParts.push(lang.toUpperCase());
		const statusBar = config.ttsEnabled
			? this.success("● ") + statusParts.join(this.dim(" · "))
			: this.dim("● disabled · ") + statusParts.join(this.dim(" · "));
		lines.push(`  ${statusBar}`);
		lines.push("");

		// Six rows:
		//   0: Enabled toggle
		//   1: Backend toggle
		//   2: Model picker (local) or read-only label (deepgram)
		//   3: Voice picker (numeric sid for local; Aura voice id for deepgram)
		//   4: Speed (cycles 0.5 / 0.75 / 1.0 / 1.25 / 1.5 / 2.0)
		//   5: Test (synthesizes "The quick brown fox …")
		const rows: { label: string; value: string; hint?: string }[] = [
			{
				label: "TTS",
				value: config.ttsEnabled ? this.success("Enabled") : this.error("Disabled"),
				hint: "toggle",
			},
			{
				label: "Backend",
				value: isLocal
					? this.success("Local (offline, sherpa-onnx)")
					: this.accent("Deepgram (cloud REST)"),
				hint: "toggle",
			},
			{
				label: "Model",
				value: isLocal
					? this.formatLocalModelLabel(config.ttsLocalModel)
					: this.dim("(deepgram backend — pick a voice instead)"),
				hint: isLocal ? "pick model ›" : undefined,
			},
			{
				label: "Voice",
				value: this.formatVoiceLabel(config),
				hint: "pick voice ›",
			},
			{
				label: "Speed",
				value: `${(config.ttsSpeed ?? 1.0).toFixed(2)}×`,
				hint: "cycle",
			},
			{
				label: "Test",
				value: this.dim("synthesize sample sentence"),
				hint: "speak now",
			},
		];

		// v7.2 — left-bar cursor + dim non-selected (HIG deference).
		const labelW = 12;
		for (let i = 0; i < rows.length; i++) {
			const r = rows[i]!;
			const isSelected = i === this.row;
			const prefix = isSelected ? `${this.accent(ICON.cursorBar)}  ` : `   `;
			const label = isSelected ? r.label.padEnd(labelW) : this.dim(r.label.padEnd(labelW));
			const hint = (isSelected && r.hint) ? this.dim(` [↵ ${r.hint}]`) : "";
			lines.push(`${prefix}${label}${r.value}${hint}`);
		}

		lines.push("");
		lines.push(this.dim("  ↵ change  ←→/Tab tabs  ↑↓ navigate  esc close"));
		lines.push(this.dim("  TTS quickstart: /voice-speak <text>  ·  /voice-speak-stop"));
		return lines;
	}

	/** Status-bar helper: format the current model+voice as one short string. */
	private formatActiveModelOrVoice(config: VoiceConfig): string {
		const isLocal = (config.ttsBackend ?? "local") === "local";
		if (isLocal) {
			const modelId = config.ttsLocalModel ?? "kitten-nano-en-v0_2";
			const sid = typeof config.ttsLocalVoiceId === "number" ? config.ttsLocalVoiceId : 0;
			// Lazy lookup — keep status compact, full label is in the rows below.
			const model = TTS_LOCAL_MODELS_REF.find(m => m.id === modelId);
			const shortName = model?.name ?? modelId;
			const voice = model?.voices.find(v => v.sid === sid);
			return voice ? `${shortName} · ${voice.name}` : `${shortName} · sid ${sid}`;
		}
		return config.ttsDeepgramVoiceId ?? "aura-asteria-en";
	}

	private formatLocalModelLabel(id: string | undefined): string {
		const modelId = id ?? "kitten-nano-en-v0_2";
		const model = TTS_LOCAL_MODELS_REF.find(m => m.id === modelId);
		if (!model) return modelId;
		const installed = TTS_INSTALLED_CHECK_REF(modelId);
		const installedTag = installed ? this.success(" ✓") : this.warning(" ⬇ download on select");
		return `${model.name} (${model.size})${installedTag}`;
	}

	private formatVoiceLabel(config: VoiceConfig): string {
		const isLocal = (config.ttsBackend ?? "local") === "local";
		if (isLocal) {
			const modelId = config.ttsLocalModel ?? "kitten-nano-en-v0_2";
			const sid = typeof config.ttsLocalVoiceId === "number" ? config.ttsLocalVoiceId : 0;
			const model = TTS_LOCAL_MODELS_REF.find(m => m.id === modelId);
			const voice = model?.voices.find(v => v.sid === sid);
			return voice ? `${voice.name} (sid ${sid})` : `sid ${sid}`;
		}
		return config.ttsDeepgramVoiceId ?? "aura-asteria-en";
	}

	// ─── Device tab ───────────────────────────────────────────────────────

	private renderDevice(_w: number, _iw: number): string[] {
		const lines: string[] = [];
		const { device } = this.p;
		const labelW = 14;

		const gpuLabel = device.gpu.hasNvidia
			? (device.gpu.gpuName || "NVIDIA")
			: device.gpu.hasMetal ? "Apple Silicon (Metal)" : "none";

		// Hardware
		lines.push(this.dim("    Hardware"));
		const hwRows: [string, string][] = [
			["Platform", `${device.platform} ${device.arch}`],
			["RAM", `${(device.totalRamMB / 1024).toFixed(1)} GB total · ${(device.freeRamMB / 1024).toFixed(1)} GB free`],
			["CPU", `${device.cpuCores} cores — ${device.cpuModel}`],
			["GPU", gpuLabel],
		];
		if (device.gpu.vramMB) hwRows.push(["VRAM", `${device.gpu.vramMB} MB`]);
		if (device.isRaspberryPi) hwRows.push(["Raspberry Pi", device.piModel || "yes"]);
		hwRows.push(["Container", device.isContainer ? "yes" : "no"]);
		hwRows.push(["Locale", device.systemLocale]);

		for (const [label, value] of hwRows) {
			lines.push(`    ${label.padEnd(labelW)}${value}`);
		}

		// Dependencies
		lines.push("");
		lines.push(this.dim("    Dependencies"));
		const sherpaOk = this.p.isSherpaAvailable();
		lines.push(`    ${"sherpa-onnx".padEnd(labelW)}${sherpaOk ? this.success("ready") : this.success("standby — loads on first recording")}`);

		// Disk space — show "fits largest model" computation so users can
		// gauge whether a download is feasible without checking model sizes.
		const freeSpace = getFreeDiskSpace(getModelsDir());
		const largest = LOCAL_MODELS.reduce((a, b) => a.sizeBytes > b.sizeBytes ? a : b);
		const fitsLargest = freeSpace !== null && freeSpace >= largest.sizeBytes;
		const diskLabel = freeSpace !== null ? formatBytes(freeSpace) + " free" : "unknown";
		const diskWarn = freeSpace !== null && freeSpace < 500 * 1024 * 1024; // <500MB
		const fitsHint = freeSpace !== null
			? ` (largest model needs ${formatBytes(largest.sizeBytes)}${fitsLargest ? " ✓" : " ✗"})`
			: "";
		lines.push(`    ${"Disk space".padEnd(labelW)}${diskWarn ? this.warning(diskLabel + " (low)") : diskLabel}${this.dim(fitsHint)}`);

		// Downloaded models total
		const downloaded = this.p.getDownloadedModels();
		const totalMB = downloaded.reduce((sum, d) => sum + d.sizeMB, 0);
		lines.push(`    ${"Models".padEnd(labelW)}${downloaded.length} downloaded (${totalMB} MB)`);

		lines.push("");
		lines.push(this.dim("  ←→/Tab tabs  esc close"));
		return lines;
	}

	// ─── Language sub-picker ──────────────────────────────────────────────

	private renderLangPicker(_w: number, _iw: number): string[] {
		const lines: string[] = [];
		const currentCode = this.p.config.language || "en";

		lines.push(`  ${this.bold("Select language")}`);
		const cursor = this.langSearch ? this.langSearch : this.dim("type to filter…");
		lines.push(`  ${this.dim("Search:")} ${cursor}`);
		lines.push("");

		const maxVisible = 12;
		const total = this.langFiltered.length;
		const start = Math.max(0, Math.min(this.langRow - Math.floor(maxVisible / 2), total - maxVisible));
		const end = Math.min(start + maxVisible, total);

		for (let i = start; i < end; i++) {
			const lang = this.langFiltered[i]!;
			const isSelected = i === this.langRow;
			const isCurrent = lang.code === currentCode;
			const prefix = isSelected ? `${this.accent(ICON.cursorBar)}  ` : `   `;
			const text = isSelected ? this.accent(`${lang.name} (${lang.code})`) : `${lang.name} (${lang.code})`;
			const check = isCurrent ? this.success(" ✓") : "";
			lines.push(`${prefix}${text}${check}`);
		}

		if (total === 0) {
			lines.push(this.dim("    No matching languages"));
		} else if (start > 0 || end < total) {
			lines.push(this.dim(`    showing ${start + 1}–${end} of ${total}`));
		}

		lines.push("");
		lines.push(this.dim("  ↵ select  esc back  type to filter"));
		return lines;
	}

	private handleLangInput(data: string): void {
		if (matchesKey(data, Key.escape)) {
			this.sub = "main";
			return;
		}
		if (matchesKey(data, Key.up)) {
			if (this.langFiltered.length > 0) {
				this.langRow = this.langRow === 0 ? this.langFiltered.length - 1 : this.langRow - 1;
			}
			return;
		}
		if (matchesKey(data, Key.down)) {
			if (this.langFiltered.length > 0) {
				this.langRow = this.langRow === this.langFiltered.length - 1 ? 0 : this.langRow + 1;
			}
			return;
		}
		if (matchesKey(data, Key.enter)) {
			const lang = this.langFiltered[this.langRow];
			if (lang) {
				this.p.config.language = lang.code;
				this.p.saveConfig(
					this.p.config,
					this.p.config.scope === "project" ? "project" : "global",
					this.p.cwd,
				);
				if (this.p.config.backend === "local") {
					try { this.p.clearRecognizerCache(); } catch {}
				}
			}
			this.sub = "main";
			return;
		}
		if (matchesKey(data, Key.backspace)) {
			this.langSearch = this.langSearch.slice(0, -1);
			this.filterLangs();
			this.langRow = 0;
			return;
		}
		if (data.length === 1 && data >= " " && data <= "~") {
			this.langSearch += data;
			this.filterLangs();
			this.langRow = 0;
		}
	}

	// ─── Actions ──────────────────────────────────────────────────────────

	private handleSelect(tabId: TabId): void {
		if (tabId === "general") {
			const { config } = this.p;
			switch (this.row) {
				case 0: // Backend toggle
					config.backend = config.backend === "local" ? "deepgram" : "local";
					this.save();
					break;
				case 1: // Model (local) or API Key (deepgram)
					if (config.backend === "local") {
						this.tab = 1;
						this.row = 0;
						this.modelSearch = "";
						this.refreshModelView();
					}
					break;
				case 2: // Language picker
					this.openLangPicker();
					break;
				case 3: // Scope toggle
					config.scope = config.scope === "project" ? "global" : "project";
					this.save();
					break;
				case 4: // Voice toggle
					config.enabled = !config.enabled;
					this.save();
					break;
				case 5: // v7.1.3: Auto-send STT toggle
					config.autoSubmitOnSpeak = !(config.autoSubmitOnSpeak === true);
					this.save();
					break;
			}
		} else if (tabId === "models") {
			const model = this.getRowAt(this.row);
			if (model) {
				this.activateModel(model.id);
				const downloaded = new Set(this.p.getDownloadedModels().map(d => d.id));
				if (!downloaded.has(model.id)) {
					this.onClose?.({ type: "download", modelId: model.id });
					return;
				}
			}
		} else if (tabId === "downloaded") {
			const dl = this.getDownloaded();
			if (this.row < dl.length) {
				const item = dl[this.row];
				if (item) this.activateModel(item.id);
			} else {
				const handyNotImported = scanHandyModels().filter(h => !h.imported);
				const handyIdx = this.row - dl.length;
				const h = handyNotImported[handyIdx];
				if (h) {
					const result = importHandyModel(h.handyId);
					if (result.ok) this.activateModel(h.piModelId);
				}
			}
		} else if (tabId === "speak") {
			const { config } = this.p;
			switch (this.row) {
				case 0: // TTS Enabled toggle
					config.ttsEnabled = !config.ttsEnabled;
					this.save();
					break;
				case 1: // Backend toggle
					config.ttsBackend = (config.ttsBackend ?? "local") === "local" ? "deepgram" : "local";
					this.save();
					break;
				case 2: // Model picker (local only — Deepgram has no model concept)
					if ((config.ttsBackend ?? "local") === "local") {
						this.openTtsModelPicker();
					}
					break;
				case 3: // Voice picker
					this.openTtsVoicePicker();
					break;
				case 4: { // Speed cycle
					const ladder = [0.75, 1.0, 1.25, 1.5, 2.0, 0.5];
					const current = config.ttsSpeed ?? 1.0;
					const idx = ladder.findIndex(v => Math.abs(v - current) < 0.01);
					config.ttsSpeed = ladder[(idx + 1) % ladder.length];
					this.save();
					break;
				}
				case 5: { // Test — emit a special panel-close action so the
					// caller (voice.ts:openSettingsPanel) can route it to
					// /voice-speak-test without us depending on the
					// command registry from inside the panel.
					this.onClose?.({ type: "speak-test" } as PanelAction);
					return;
				}
			}
		}
	}

	/**
	 * First press arms the delete on the currently-selected downloaded model
	 * with a 1.5s window. Second press within that window commits. Different
	 * model selection or any other key aborts.
	 */
	private handleDeleteRequest(): void {
		const dl = this.getDownloaded();
		const item = dl[this.row];
		if (!item) return; // Handy import row, not deletable here

		const now = Date.now();
		if (this.deletePendingId === item.id && now <= this.deletePendingExpiresAt) {
			// Commit
			const wasActive = this.p.config.localModel === item.id;
			this.p.deleteModel(item.id);
			if (wasActive) {
				try { this.p.clearRecognizerCache(); } catch {}
				const remaining = this.p.getDownloadedModels();
				this.p.config.localModel = remaining.length > 0 ? remaining[0]!.id : undefined;
				this.save();
			}
			this.deletePendingId = null;
			this.deletePendingExpiresAt = 0;
			this.row = Math.max(0, Math.min(this.row, dl.length - 2));
		} else {
			// Arm
			this.deletePendingId = item.id;
			this.deletePendingExpiresAt = now + VoiceSettingsPanel.DELETE_CONFIRM_MS;
		}
	}

	private activateModel(modelId: string): void {
		const { config } = this.p;
		if (config.localModel !== modelId) {
			try { this.p.clearRecognizerCache(); } catch {}
		}
		config.localModel = modelId;
		config.backend = "local";
		config.localEndpoint = undefined;
		this.save();
		this.rebuildModels();
	}

	private openLangPicker(): void {
		const { config } = this.p;
		if (config.backend === "local" && config.localModel) {
			const { languages, englishOnly } = getLanguagesForLocalModel(config.localModel);
			if (englishOnly) return; // Single language — nothing to pick
			this.langList = languages;
		} else {
			this.langList = this.p.deepgramLanguages;
		}
		this.langSearch = "";
		this.langFiltered = this.langList;
		this.langRow = 0;
		const idx = this.langList.findIndex(l => l.code === config.language);
		if (idx >= 0) this.langRow = idx;
		this.sub = "lang-picker";
	}

	private save(): void {
		const { config, cwd } = this.p;
		this.p.saveConfig(config, config.scope === "project" ? "project" : "global", cwd);
	}

	// ─── TTS Model picker ──────────────────────────────────────────────────

	/** Lazy chassis getter — created on first model picker open. */
	private getTtsModelChassis(): PickerChassis<TtsLocalModelInfo> {
		if (!this.ttsModelChassis) {
			this.ttsModelChassis = new PickerChassis<TtsLocalModelInfo>();
			this.ttsModelChassis.setRows(buildTtsModelPickerRows(TTS_LOCAL_MODELS_REF));
		}
		return this.ttsModelChassis;
	}

	private openTtsModelPicker(): void {
		const chassis = this.getTtsModelChassis();
		chassis.clearSearch();
		const currentId = this.p.config.ttsLocalModel ?? "kitten-nano-en-v0_2";
		const current = TTS_LOCAL_MODELS_REF.find(m => m.id === currentId);
		if (current) chassis.selectValue(current);
		this.sub = "tts-model-picker";
	}

	private renderTtsModelPicker(w: number, iw: number): string[] {
		const lines: string[] = [];
		const chassis = this.getTtsModelChassis();
		const currentId = this.p.config.ttsLocalModel ?? "kitten-nano-en-v0_2";

		lines.push(`  ${this.bold("Pick TTS model")}`);
		const query = chassis.getQuery();
		const cursor = query ? query : this.dim("type to filter…");
		lines.push(`  ${this.dim("Search:")} ${cursor}`);
		lines.push("");

		// v7.1: width-tier compact mode (§10 mid-tier 60..79 drops headings)
		const compact = w < 80;
		const view = chassis.view({ maxVisible: 12, compact });

		if (view.kind === "empty") {
			lines.push(this.dim(`    No matches for "${query}".`));
			lines.push("");
			lines.push(this.dim("  esc back  bksp clear search"));
			return lines;
		}

		const nameW = Math.min(28, Math.max(18, iw - 32));
		const selectedValue = chassis.selected();
		for (const r of view.rows) {
			if (r.kind === "heading") {
				// Compact mode never emits headings; in wide mode they
				// render as a dim label with a leading chevron.
				lines.push(`  ${this.dim(`${ICON.middot} ${r.label}`)}`);
				continue;
			}
			const m = r.value;
			const isSelected = m === selectedValue;
			const isCurrent = m.id === currentId;
			const installed = TTS_INSTALLED_CHECK_REF(m.id);
			// v7.2 — selected row has a thin accent left bar; non-selected
			// rows are dim. HIG "deference": chrome is subtle, content
			// hierarchy comes through dim/full-saturation contrast.
			const prefix = isSelected
				? `${this.accent(ICON.cursorBar)}  `
				: `   `;
			const name = isSelected ? this.accent(m.name) : this.dim(m.name);
			const namePad = m.name.length < nameW ? " ".repeat(nameW - m.name.length) : "";
			const size = this.dim(m.size.padStart(8));
			const langs = this.dim(m.languages.length > 1
				? `${m.languages.length} langs`.padEnd(13)
				: m.languages[0]!.padEnd(13));
			// v7.2 — colored-dot status badges. Replaces plain "active"/
			// "ready"/"download" words with `● active` / `● ready` /
			// `○ download` / `✗ broken` patterns. Charm/gum convention.
			const status = m.incompatible
				? this.warning(`${ICON.checkFail} broken`)
				: isCurrent
					? this.success(`${ICON.bulletActive} active`)
					: installed
						? this.success(`${ICON.bulletActive} ready`)
						: this.warning(`${ICON.bulletInactive} download`);
			lines.push(`${prefix}${name}${namePad} ${size}  ${langs} ${status}`);
			if (isSelected) {
				if (m.incompatible) {
					lines.push(`        ${this.warning(m.incompatible)}`);
				} else {
					lines.push(`        ${this.dim(m.notes)}`);
				}
			}
		}

		if (view.viewportStart > 0 || view.viewportEnd < view.totalSelectable) {
			lines.push(this.dim(`    showing ${view.viewportStart + 1}–${view.viewportEnd} of ${view.totalSelectable}`));
		}
		lines.push("");
		const sel_m = selectedValue;
		if (sel_m) {
			const installed = TTS_INSTALLED_CHECK_REF(sel_m.id);
			const enterHint = installed ? "activate" : `download (${sel_m.size}) + activate`;
			lines.push(this.dim(`  ${ICON.arrowRight} ${enterHint}  esc back  type to filter`));
		} else {
			lines.push(this.dim("  esc back"));
		}
		return lines;
	}

	private handleTtsModelInput(data: string): void {
		const chassis = this.getTtsModelChassis();
		if (matchesKey(data, Key.escape)) {
			this.sub = "main";
			return;
		}
		if (matchesKey(data, Key.up)) {
			chassis.moveUp();
			return;
		}
		if (matchesKey(data, Key.down)) {
			chassis.moveDown();
			return;
		}
		if (matchesKey(data, Key.enter)) {
			const m = chassis.selected();
			if (!m) return;
			// v7.1.2 — refuse to activate models flagged as incompatible
			// with the installed sherpa-onnx runtime. The picker still
			// lists them (so users can see future-fix candidates) but
			// pressing enter on a broken model is a no-op.
			if (m.incompatible) return;
			this.p.config.ttsLocalModel = m.id;
			// Reset voice id when model changes — preserve sid 0 default.
			this.p.config.ttsLocalVoiceId = m.defaultSid;
			this.save();
			this.sub = "main";
			// If model isn't installed, signal to the caller via panel
			// close so voice.ts's openSettingsPanel post-close handler can
			// run ensureTtsModelInstalled (now via the v7.1 install widget).
			if (!TTS_INSTALLED_CHECK_REF(m.id)) {
				this.onClose?.({ type: "tts-install", modelId: m.id });
			}
			return;
		}
		if (matchesKey(data, Key.backspace)) {
			chassis.backspaceSearch();
			return;
		}
		if (data.length === 1 && data >= " " && data <= "~") {
			chassis.appendSearchChar(data);
		}
	}

	// ─── TTS Voice picker ──────────────────────────────────────────────────

	/**
	 * v7.1 — native-script language label for a voice picker row, or a
	 * romanized fallback for ar/hi/unknown. Returns empty string when
	 * no language tag is available (we don't show "Unknown"). The
	 * gender word is intentionally NOT rendered here — the existing
	 * `(male)`/`(female)` meta block already shows it.
	 */
	private formatVoiceNativeLabel(language: string | undefined, _gender: string | undefined): string {
		if (!language) return "";
		// BCP-47 base tag — drop region/script subtags. e.g. "zh-Hant-TW" → "zh".
		const base = language.split("-")[0]!.toLowerCase();
		const native = localeLabel(base);
		if (native) {
			// Only show if it fits in 8 visual columns — keeps the row tidy.
			if (visualWidth(native.nativeName) <= 8) return native.nativeName;
			return base.toUpperCase();
		}
		// Romanized fallback (ar / hi / unknown).
		return formatRomanizedLabel(base, undefined);
	}

	private getCurrentVoiceCatalog(): { id: string | number; label: string; meta?: string; language?: string }[] {
		const { config } = this.p;
		const isLocal = (config.ttsBackend ?? "local") === "local";
		if (isLocal) {
			const modelId = config.ttsLocalModel ?? "kitten-nano-en-v0_2";
			const model = TTS_LOCAL_MODELS_REF.find(m => m.id === modelId);
			if (!model) return [];
			// v7.1: thread the model's primary language so the picker
			// can render native-script labels via `localeLabel()`.
			const language = model.languages[0];
			return model.voices.map((v: TtsVoice) => ({
				id: v.sid,
				label: v.name,
				meta: v.gender,
				language,
			}));
		}
		// Deepgram: filter Aura voices by current language for relevance.
		const lang = config.ttsLanguage || config.language || "en";
		const filtered = filterDeepgramVoicesByLanguage(lang);
		const list = filtered.length > 0 ? filtered : DEEPGRAM_TTS_VOICES;
		return list.map(v => ({ id: v.id, label: v.name, meta: v.gender, language: v.language }));
	}

	private getFilteredTtsVoices(): { id: string | number; label: string; meta?: string; language?: string }[] {
		const all = this.getCurrentVoiceCatalog();
		const q = this.ttsVoiceSearch.trim().toLowerCase();
		if (!q) return all;
		return all.filter(v => `${v.label} ${v.meta ?? ""} ${v.id} ${v.language ?? ""}`.toLowerCase().includes(q));
	}

	private openTtsVoicePicker(): void {
		this.ttsVoiceSearch = "";
		const all = this.getCurrentVoiceCatalog();
		const { config } = this.p;
		const isLocal = (config.ttsBackend ?? "local") === "local";
		const currentId: string | number = isLocal
			? (typeof config.ttsLocalVoiceId === "number" ? config.ttsLocalVoiceId : 0)
			: (config.ttsDeepgramVoiceId ?? "aura-asteria-en");
		const idx = all.findIndex(v => v.id === currentId);
		this.ttsVoiceRow = idx >= 0 ? idx : 0;
		this.sub = "tts-voice-picker";
	}

	private renderTtsVoicePicker(_w: number, _iw: number): string[] {
		const lines: string[] = [];
		const filtered = this.getFilteredTtsVoices();
		const { config } = this.p;
		const isLocal = (config.ttsBackend ?? "local") === "local";
		const currentId: string | number = isLocal
			? (typeof config.ttsLocalVoiceId === "number" ? config.ttsLocalVoiceId : 0)
			: (config.ttsDeepgramVoiceId ?? "aura-asteria-en");

		lines.push(`  ${this.bold(isLocal ? "Pick local voice" : "Pick Deepgram voice")}`);
		const cursor = this.ttsVoiceSearch ? this.ttsVoiceSearch : this.dim("type to filter…");
		lines.push(`  ${this.dim("Search:")} ${cursor}`);
		lines.push("");

		if (filtered.length === 0) {
			lines.push(this.dim("    No matching voices."));
			lines.push("");
			lines.push(this.dim("  esc back  type to filter"));
			return lines;
		}

		const maxVisible = 12;
		const total = filtered.length;
		const sel = Math.min(this.ttsVoiceRow, total - 1);
		let start = Math.max(0, sel - Math.floor(maxVisible / 2));
		const end = Math.min(start + maxVisible, total);
		if (end - start < maxVisible) start = Math.max(0, end - maxVisible);

		for (let i = start; i < end; i++) {
			const v = filtered[i]!;
			const isSelected = i === sel;
			const isCurrent = v.id === currentId;
			// v7.2: thin accent left bar + dim non-selected rows.
			const prefix = isSelected ? `${this.accent(ICON.cursorBar)}  ` : `   `;
			const idStr = typeof v.id === "number" ? `sid ${v.id}` : v.id;
			const text = isSelected ? this.accent(v.label) : this.dim(v.label);
			const meta = v.meta ? this.dim(` (${v.meta})`) : "";
			const idTag = this.dim(` ${ICON.middot} ${idStr}`);
			const check = isCurrent ? this.success(` ${ICON.checkOk}`) : "";
			// v7.1: append native-script language label when available
			// (omits ar/hi per `ui-locale-labels.ts` to keep widths sane).
			const langLabel = this.formatVoiceNativeLabel(v.language, v.meta);
			const langSuffix = langLabel ? this.dim(`  ${ICON.bullet} ${langLabel}`) : "";
			lines.push(`${prefix}${text}${meta}${idTag}${check}${langSuffix}`);
		}

		if (start > 0 || end < total) {
			lines.push(this.dim(`    showing ${start + 1}–${end} of ${total}`));
		}
		lines.push("");
		lines.push(this.dim("  ↵ select  esc back  type to filter"));
		return lines;
	}

	private handleTtsVoiceInput(data: string): void {
		if (matchesKey(data, Key.escape)) {
			this.sub = "main";
			return;
		}
		const filtered = this.getFilteredTtsVoices();
		if (matchesKey(data, Key.up)) {
			if (filtered.length > 0) {
				this.ttsVoiceRow = this.ttsVoiceRow === 0 ? filtered.length - 1 : this.ttsVoiceRow - 1;
			}
			return;
		}
		if (matchesKey(data, Key.down)) {
			if (filtered.length > 0) {
				this.ttsVoiceRow = this.ttsVoiceRow === filtered.length - 1 ? 0 : this.ttsVoiceRow + 1;
			}
			return;
		}
		if (matchesKey(data, Key.enter)) {
			const v = filtered[this.ttsVoiceRow];
			if (!v) return;
			const isLocal = (this.p.config.ttsBackend ?? "local") === "local";
			if (isLocal && typeof v.id === "number") {
				this.p.config.ttsLocalVoiceId = v.id;
			} else if (!isLocal && typeof v.id === "string") {
				this.p.config.ttsDeepgramVoiceId = v.id;
			}
			this.save();
			this.sub = "main";
			return;
		}
		if (matchesKey(data, Key.backspace)) {
			this.ttsVoiceSearch = this.ttsVoiceSearch.slice(0, -1);
			this.ttsVoiceRow = 0;
			return;
		}
		if (data.length === 1 && data >= " " && data <= "~") {
			this.ttsVoiceSearch += data;
			this.ttsVoiceRow = 0;
		}
	}

	// ─── Helpers ──────────────────────────────────────────────────────────

	private getRowCount(tabId: TabId): number {
		switch (tabId) {
			case "general": return 6;
			case "models": return this.modelSelectableIdx.length;
			case "downloaded": {
				const dl = this.getDownloaded().length;
				const handy = scanHandyModels().filter(h => !h.imported).length;
				return dl + handy;
			}
			case "speak": return 6;
			case "device": return 0;
		}
	}

	private getRowAt(row: number): ModelRow | undefined {
		const flatIdx = this.modelSelectableIdx[row];
		if (flatIdx === undefined) return undefined;
		const item = this.modelRowsFlat[flatIdx];
		return item && item.kind === "row" ? item.row : undefined;
	}

	/**
	 * Build the grouped model view: top picks (recommended for current device)
	 * first, then by family/language. Search filters all groups together;
	 * empty groups are dropped so the user never sees an orphaned heading.
	 */
	private rebuildModels(): void {
		const enriched: ModelRow[] = LOCAL_MODELS.map(m => ({
			...m,
			fitness: this.p.getModelFitness(m, this.p.device) as ModelFitness,
		}));
		this.modelGroups = groupModels(enriched);
		this.refreshModelView();
	}

	/** Apply the search query and rebuild the flat row list. */
	private refreshModelView(): void {
		const q = this.modelSearch.trim().toLowerCase();
		const flat: typeof this.modelRowsFlat = [];
		const selectable: number[] = [];

		for (const group of this.modelGroups) {
			const rows = q
				? group.rows.filter(m => `${m.name} ${m.id} ${m.notes} ${m.langSupport}`.toLowerCase().includes(q))
				: group.rows;
			if (rows.length === 0) continue;
			flat.push({ kind: "heading", group });
			for (const row of rows) {
				selectable.push(flat.length);
				flat.push({ kind: "row", row });
			}
		}

		this.modelRowsFlat = flat;
		this.modelSelectableIdx = selectable;
		this.row = Math.min(this.row, Math.max(0, selectable.length - 1));
	}

	private filterLangs(): void {
		if (!this.langSearch) {
			this.langFiltered = this.langList;
		} else {
			const q = this.langSearch.toLowerCase();
			this.langFiltered = this.langList.filter(l =>
				`${l.name} ${l.code}`.toLowerCase().includes(q),
			);
		}
		this.langRow = Math.min(this.langRow, Math.max(0, this.langFiltered.length - 1));
	}

	private getDownloaded(): { id: string; name: string; sizeMB: number; isCurrent: boolean }[] {
		const currentId = this.p.config.localModel || "parakeet-v3";
		return this.p.getDownloadedModels().map(d => ({
			...d,
			name: LOCAL_MODELS.find(m => m.id === d.id)?.name || d.id,
			isCurrent: d.id === currentId,
		}));
	}

	private getLangDisplay(): string {
		const code = this.p.config.language || "en";
		const allLangs = [...this.p.deepgramLanguages];
		for (const m of LOCAL_MODELS) {
			const { languages } = getLanguagesForLocalModel(m.id);
			allLangs.push(...languages);
		}
		const entry = allLangs.find(l => l.code === code);
		return entry ? `${entry.name} (${code})` : code;
	}

	/** Single-axis rating bar shown only on the selected (expanded) row. */
	private ratingBar(value: 1 | 2 | 3 | 4 | 5, label: string): string {
		const filled = "●".repeat(value);
		const empty = "○".repeat(5 - value);
		return this.dim(label) + " " + filled + this.dim(empty);
	}
}

// ─── Free helpers ──────────────────────────────────────────────────────────

/**
 * Compact language-coverage hint for a model row in the Models tab.
 * Prefer human readable scope ("English", "25 langs", "Russian") over the
 * raw `langSupport` enum — users care about coverage, not internal tags.
 */
function formatLangHint(m: LocalModelInfo): string {
	switch (m.langSupport) {
		case "whisper": return "57 langs";
		case "english-only": return "English";
		case "parakeet-multi": return "25 langs";
		case "sensevoice": return "zh/en/ja/ko";
		case "russian-only": return "Russian";
		case "single-ar": return "Arabic";
		case "single-zh": return "Chinese";
		case "single-ja": return "Japanese";
		case "single-ko": return "Korean";
		case "single-uk": return "Ukrainian";
		case "single-vi": return "Vietnamese";
		case "single-es": return "Spanish";
	}
	return "";
}

/** Short fitness label shown in the rightmost cell on the Models tab. */
function formatFitness(f: ModelFitness): string {
	switch (f) {
		case "recommended": return "recommended";
		case "compatible": return "compatible";
		case "warning": return "may be slow";
		case "incompatible": return "too large";
	}
}

/**
 * Group models for the Models tab.
 *   - "Top picks for your device" — fitness === "recommended", capped at 4
 *   - "Whisper" — all whisper-* (57-language family)
 *   - "Moonshine" — moonshine-* (edge / fast English + variants)
 *   - "Specialist" — single-language and zh/en/ja/ko models
 *   - "All multilingual" — Parakeet TDT family
 *
 * Models can appear under "Top picks" AND their family group; that's
 * intentional — top picks is the user's fast path, family groups are the
 * "I want to compare alternatives" path.
 */
function groupModels(rows: ModelRow[]): ModelGroup[] {
	const byFamily = (predicate: (r: ModelRow) => boolean) => rows.filter(predicate);

	const topPicks = rows.filter(r => r.fitness === "recommended").slice(0, 4);
	const parakeet = byFamily(r => r.id.startsWith("parakeet-"));
	const whisper = byFamily(r => r.id.startsWith("whisper-"));
	const moonshine = byFamily(r => r.id.startsWith("moonshine-"));
	const specialist = byFamily(r =>
		!r.id.startsWith("parakeet-") &&
		!r.id.startsWith("whisper-") &&
		!r.id.startsWith("moonshine-"),
	);

	const groups: ModelGroup[] = [];
	if (topPicks.length > 0) {
		groups.push({
			heading: "Top picks for your device",
			subtitle: "fitness-ranked",
			rows: topPicks,
		});
	}
	if (parakeet.length > 0) {
		groups.push({
			heading: "Parakeet (multilingual)",
			subtitle: "NVIDIA NeMo · TDT",
			rows: parakeet,
		});
	}
	if (whisper.length > 0) {
		groups.push({
			heading: "Whisper (broad coverage)",
			subtitle: "OpenAI · 57 languages",
			rows: whisper,
		});
	}
	if (moonshine.length > 0) {
		groups.push({
			heading: "Moonshine (edge / fast)",
			subtitle: "Useful Sensors · low latency",
			rows: moonshine,
		});
	}
	if (specialist.length > 0) {
		groups.push({
			heading: "Specialist",
			subtitle: "best-in-class for one or a few languages",
			rows: specialist,
		});
	}
	return groups;
}
