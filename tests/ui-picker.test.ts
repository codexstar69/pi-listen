import { describe, expect, test } from "bun:test";
import { PickerChassis, type PickerRow } from "../extensions/voice/ui-picker";

interface VoiceItem { id: string; name: string }

const TIER0: PickerRow<VoiceItem>[] = [
	{ kind: "heading", label: "Tier 0 — Recommended" },
	{ kind: "data", value: { id: "kitten", name: "Kitten Nano" }, searchKey: "kitten nano en" },
	{ kind: "data", value: { id: "kokoro", name: "Kokoro" }, searchKey: "kokoro multi" },
];

const TIER1: PickerRow<VoiceItem>[] = [
	{ kind: "heading", label: "English Piper" },
	{ kind: "data", value: { id: "piper-en1", name: "Piper EN-Lessac" }, searchKey: "piper english lessac" },
	{ kind: "data", value: { id: "piper-en2", name: "Piper EN-Ryan" }, searchKey: "piper english ryan" },
];

const TIER2: PickerRow<VoiceItem>[] = [
	{ kind: "heading", label: "Heavyweight" },
	{ kind: "data", value: { id: "matcha", name: "Matcha" }, searchKey: "matcha multi" },
];

const ALL: PickerRow<VoiceItem>[] = [...TIER0, ...TIER1, ...TIER2];

describe("PickerChassis — heading-aware navigation", () => {
	test("moveDown skips headings", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		// Cursor starts at first selectable (kitten)
		expect(p.selected()?.id).toBe("kitten");
		p.moveDown();
		expect(p.selected()?.id).toBe("kokoro");
		p.moveDown();
		expect(p.selected()?.id).toBe("piper-en1"); // heading skipped
		p.moveDown();
		expect(p.selected()?.id).toBe("piper-en2");
		p.moveDown();
		expect(p.selected()?.id).toBe("matcha"); // heading skipped
	});
	test("moveUp wraps from first to last selectable", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.moveUp();
		expect(p.selected()?.id).toBe("matcha");
	});
	test("moveDown wraps from last to first selectable", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.selectValue({ id: "matcha", name: "Matcha" } as VoiceItem); // by reference won't match; use ref
		// Move to matcha by repeated downs:
		for (let i = 0; i < 4; i++) p.moveDown();
		expect(p.selected()?.id).toBe("matcha");
		p.moveDown();
		expect(p.selected()?.id).toBe("kitten");
	});
});

describe("PickerChassis — search filtering with heading skipping", () => {
	test("query that matches all groups still shows all groups", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.setSearch("multi");
		const view = p.view({ maxVisible: 12, compact: false });
		expect(view.kind).toBe("list");
		if (view.kind !== "list") return;
		// kokoro + matcha both match "multi"; their headings come along
		const labels = view.rows.map(r => (r.kind === "heading" ? `H:${r.label}` : `D:${r.value.id}`));
		expect(labels).toContain("D:kokoro");
		expect(labels).toContain("D:matcha");
		expect(labels).toContain("H:Tier 0 — Recommended");
		expect(labels).toContain("H:Heavyweight");
		// English Piper heading should NOT appear (no matches under it)
		expect(labels).not.toContain("H:English Piper");
	});
	test("query that matches no rows returns empty view", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.setSearch("zzznever");
		const view = p.view({ maxVisible: 12, compact: false });
		expect(view.kind).toBe("empty");
	});
	test("compact mode drops headings entirely", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		const view = p.view({ maxVisible: 12, compact: true });
		expect(view.kind).toBe("list");
		if (view.kind !== "list") return;
		expect(view.rows.every(r => r.kind === "data")).toBe(true);
	});
	test("heading is not duplicated when group has multiple matches", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.setSearch("piper"); // both piper rows match
		const view = p.view({ maxVisible: 12, compact: false });
		expect(view.kind).toBe("list");
		if (view.kind !== "list") return;
		const headingHits = view.rows.filter(r => r.kind === "heading" && r.label === "English Piper").length;
		expect(headingHits).toBe(1);
	});
});

describe("PickerChassis — cursor restoration", () => {
	test("cursor resets to first match when search applies", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.moveDown(); // cursor on kokoro
		p.setSearch("piper");
		expect(p.selected()?.id).toBe("piper-en1"); // first match
	});
	test("cursor restores to previously-active row when search clears", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.moveDown(); // cursor on kokoro
		p.setSearch("piper"); // cursor jumps to piper-en1
		p.setSearch(""); // clear → cursor should restore to kokoro
		expect(p.selected()?.id).toBe("kokoro");
	});
	test("cursor falls back to first when restored value is no longer present", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.moveDown(); // kokoro
		p.setSearch("piper");
		// Remove kokoro from the data: only headings + kitten + matcha remain.
		p.setRows([
			{ kind: "heading", label: "Tier 0 — Recommended" },
			{ kind: "data", value: { id: "kitten", name: "Kitten Nano" }, searchKey: "kitten nano en" },
			{ kind: "heading", label: "Heavyweight" },
			{ kind: "data", value: { id: "matcha", name: "Matcha" }, searchKey: "matcha multi" },
		]);
		p.setSearch("");
		expect(p.selected()?.id).toBe("kitten"); // first selectable
	});
});

describe("PickerChassis — input helpers", () => {
	test("appendSearchChar / backspaceSearch move query forward + back", () => {
		const p = new PickerChassis<VoiceItem>();
		p.setRows(ALL);
		p.appendSearchChar("p");
		p.appendSearchChar("i");
		expect(p.getQuery()).toBe("pi");
		p.backspaceSearch();
		expect(p.getQuery()).toBe("p");
		p.backspaceSearch();
		expect(p.getQuery()).toBe("");
		// Backspace on empty is a no-op.
		expect(() => p.backspaceSearch()).not.toThrow();
	});
});
