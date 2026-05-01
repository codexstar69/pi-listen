/**
 * Shared picker chassis (§3 of v7.1 plan).
 *
 * Three pickers (Language / Models / Voice) currently duplicate the same
 * state machine in `settings-panel.ts`. This chassis encapsulates the
 * common behavior: heading-aware navigation, search filtering with empty
 * state, cursor preservation across filter changes, and a width-tier
 * compact mode. Pickers feed it data; the chassis owns navigation.
 *
 * Contract (§3):
 *   1. Headings are non-selectable. ↑↓ skip them.
 *   2. Search filtering also skips headings: a heading appears iff at
 *      least one row under it matches; otherwise drop the entire group.
 *   3. Cursor restoration: when the search query changes, the cursor
 *      moves to the first selectable row of the new view. When the
 *      search clears, the cursor returns to the previously-active row
 *      if visible, else first selectable.
 *   4. Empty state: render-aware — `getViewModel()` returns
 *      `{ kind: "empty" }` when no data rows match.
 *   5. Page bounds: cursor wraps top-to-bottom only across selectable rows.
 *   6. Width fallback: when compact mode is true, headings are dropped
 *      from the view (the picker may still show group separators
 *      elsewhere) so narrow terminals see only the data rows.
 */

/** A row in a picker — either a heading (non-selectable) or a data row. */
export type PickerRow<T> =
	| { readonly kind: "heading"; readonly label: string }
	| { readonly kind: "data"; readonly value: T; readonly searchKey: string };

/** Output passed to the renderer. Either an empty result or a viewport slice. */
export type PickerView<T> =
	| { readonly kind: "empty"; readonly query: string }
	| {
			readonly kind: "list";
			readonly rows: ReadonlyArray<PickerRow<T>>;
			readonly viewportStart: number;
			readonly viewportEnd: number;
			readonly totalSelectable: number;
			readonly selectedIndex: number;
			/** Index INTO `rows` of the currently-selected data row (≥0). */
			readonly cursorRowIndex: number;
	  };

/** Per-call render input. */
export interface PickerRenderInput {
	/** Visible row budget for the body — usually 12 in 24-line overlays. */
	readonly maxVisible: number;
	/** True for narrow-terminal compact mode (drops headings from view). */
	readonly compact: boolean;
}

/**
 * State machine. Pickers create one of these per open session and call:
 *   - `setRows(rows)` whenever the data source changes (e.g. after install)
 *   - `setSearch(q)` on every keystroke
 *   - `moveUp()` / `moveDown()` for navigation
 *   - `selected()` to read the currently-highlighted data value
 *   - `view(input)` to compute the renderable slice
 *
 * The chassis owns: search query, cursor index (over selectable rows
 * only), viewport scrolling.
 */
export class PickerChassis<T> {
	private rows: ReadonlyArray<PickerRow<T>> = [];
	private query = "";
	/** Cursor index INTO the filtered selectable subset. */
	private cursor = 0;
	/** Sticky pre-search cursor — used to restore on search clear. */
	private preSearchValue: T | null = null;

	setRows(rows: ReadonlyArray<PickerRow<T>>): void {
		this.rows = rows;
		this.clampCursor();
	}

	getQuery(): string {
		return this.query;
	}

	setSearch(q: string): void {
		const wasEmpty = this.query.length === 0;
		const isEmpty = q.length === 0;
		if (wasEmpty && !isEmpty) {
			// Entering search — remember current selection so we can
			// restore on clear.
			this.preSearchValue = this.selected();
		}
		this.query = q;
		this.cursor = 0; // every search change resets to first match
		if (isEmpty && this.preSearchValue != null) {
			// Restore cursor to previously-active value if still present.
			const filtered = this.filteredSelectable();
			const idx = filtered.findIndex(r => r.value === this.preSearchValue);
			if (idx >= 0) this.cursor = idx;
			this.preSearchValue = null;
		}
	}

	appendSearchChar(ch: string): void {
		this.setSearch(this.query + ch);
	}

	backspaceSearch(): void {
		if (this.query.length === 0) return;
		this.setSearch(this.query.slice(0, -1));
	}

	clearSearch(): void {
		this.setSearch("");
	}

	moveUp(): void {
		const n = this.filteredSelectable().length;
		if (n === 0) return;
		this.cursor = this.cursor === 0 ? n - 1 : this.cursor - 1;
	}

	moveDown(): void {
		const n = this.filteredSelectable().length;
		if (n === 0) return;
		this.cursor = this.cursor === n - 1 ? 0 : this.cursor + 1;
	}

	/** Move cursor to the first selectable whose `value === target`. */
	selectValue(target: T): void {
		const filtered = this.filteredSelectable();
		const idx = filtered.findIndex(r => r.value === target);
		if (idx >= 0) this.cursor = idx;
	}

	/** Currently-highlighted data value, or null when no rows match. */
	selected(): T | null {
		const filtered = this.filteredSelectable();
		return filtered[this.cursor]?.value ?? null;
	}

	/** Compute the renderable view: heading-aware, viewport-windowed. */
	view(input: PickerRenderInput): PickerView<T> {
		const filteredAll = this.filteredView(input.compact);
		const selectable = filteredAll.filter((r): r is { kind: "data"; value: T; searchKey: string } => r.kind === "data");
		if (selectable.length === 0) {
			return { kind: "empty", query: this.query };
		}
		const sel = Math.min(this.cursor, selectable.length - 1);
		const selectedValue = selectable[sel]!.value;
		const cursorRowIndex = filteredAll.findIndex(r => r.kind === "data" && r.value === selectedValue);

		// Center the viewport on the cursor.
		const total = filteredAll.length;
		let start = Math.max(0, cursorRowIndex - Math.floor(input.maxVisible / 2));
		let end = Math.min(start + input.maxVisible, total);
		if (end - start < input.maxVisible) start = Math.max(0, end - input.maxVisible);

		return {
			kind: "list",
			rows: filteredAll.slice(start, end),
			viewportStart: start,
			viewportEnd: end,
			totalSelectable: selectable.length,
			selectedIndex: sel,
			cursorRowIndex,
		};
	}

	private clampCursor(): void {
		const n = this.filteredSelectable().length;
		if (n === 0) this.cursor = 0;
		else if (this.cursor >= n) this.cursor = n - 1;
	}

	private rowMatches(r: PickerRow<T>): boolean {
		if (r.kind === "heading") return false;
		if (this.query.length === 0) return true;
		return r.searchKey.toLowerCase().includes(this.query.toLowerCase());
	}

	/** Filtered subset, headings retained when at least one child row matches. */
	private filteredView(compact: boolean): ReadonlyArray<PickerRow<T>> {
		const out: PickerRow<T>[] = [];
		// Walk groups: stash the most recent heading; emit it inline at
		// first matching child, then clear so subsequent matches in the
		// same group don't re-emit. In compact mode, never emit
		// headings (narrow terminals get a flat list).
		let pendingHeading: PickerRow<T> | null = null;
		for (const r of this.rows) {
			if (r.kind === "heading") {
				pendingHeading = r;
				continue;
			}
			if (this.rowMatches(r)) {
				if (pendingHeading && !compact) {
					out.push(pendingHeading);
				}
				pendingHeading = null;
				out.push(r);
			}
		}
		return out;
	}

	private filteredSelectable(): ReadonlyArray<{ kind: "data"; value: T; searchKey: string }> {
		const out: { kind: "data"; value: T; searchKey: string }[] = [];
		for (const r of this.rows) {
			if (r.kind === "data" && this.rowMatches(r)) out.push(r);
		}
		return out;
	}
}
