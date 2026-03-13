/**
 * Gelatinous slime pet — adapted for Pi voice widget.
 *
 * Renders a soft-body metaball slime creature using half-block characters
 * and 24-bit ANSI color. Driven by audioLevel instead of keyboard input.
 *
 * Based on /Users/codex/gelatinous_slime_pet.ts
 */

// --- Configuration ---
// All coordinates are in native widget space (no scaling from 80×72).
const SW = 30;   // Widget pixel width (characters)
const SH = 8;    // Widget pixel height (character rows, each = 2 logical pixels)
const LOGICAL_H = SH * 2; // 16 logical pixels tall

// Center of the widget
const CX = SW / 2;   // 15
const CY = LOGICAL_H / 2; // 8

// --- Slime State ---
let time = 0;
let mood: "idle" | "talking" | "sleeping" | "excited" = "idle";
let blinkFade = 0;
let nextBlinkAt = 2 + Math.random() * 3;
let talkTimer = 0;

// Flavor (Catppuccin Mocha - Mauve)
const flavor = { r: 203, g: 166, b: 247, blushR: 245, blushG: 130, blushB: 186, mouthR: 180, mouthG: 120, mouthB: 200 };

// Metablobs in native widget coords: 0: Body, 1: Left Ear, 2: Right Ear
interface Blob { x: number; y: number; rx: number; ry: number; rot: number; }
const blobs: Blob[] = [
	{ x: CX, y: CY + 1, rx: 8, ry: 5, rot: 0 },         // Body — centered, slightly below mid
	{ x: CX - 5, y: CY - 4, rx: 2, ry: 4, rot: -0.4 },  // Left ear
	{ x: CX + 5, y: CY - 4, rx: 2, ry: 4, rot: 0.4 },   // Right ear
];

const vels: Blob[] = Array.from({ length: 3 }, () => ({ x: 0, y: 0, rx: 0, ry: 0, rot: 0 }));

// Output buffer
const outColors = new Uint8Array(SW * LOGICAL_H * 3);

// --- Math ---
function getDensity(x: number, y: number): number {
	let d = 0;
	for (let i = 0; i < blobs.length; i++) {
		const b = blobs[i];
		let dx = x - b.x;
		let dy = y - b.y;
		// Quick bounding box skip — if far from blob, contribution is negligible
		const maxR = Math.max(b.rx, b.ry) * 2.5;
		if (dx > maxR || dx < -maxR || dy > maxR || dy < -maxR) continue;
		if (b.rot !== 0) {
			const s = Math.sin(b.rot);
			const c = Math.cos(b.rot);
			const nx = dx * c - dy * s;
			const ny = dx * s + dy * c;
			dx = nx; dy = ny;
		}
		const distSq = Math.max(0.0001, (dx * dx) / (b.rx * b.rx) + (dy * dy) / (b.ry * b.ry));
		d += 1.0 / distSq;
	}
	return d;
}

function updatePhysics(dt: number, audioLevel: number) {
	time += dt;
	if (talkTimer > 0) talkTimer -= dt;

	// Auto-mood from audio level
	if (audioLevel > 0.35) {
		mood = "excited";
		talkTimer = 0.5;
	} else if (audioLevel > 0.08) {
		mood = "talking";
		talkTimer = 0.5;
	} else if (talkTimer <= 0) {
		mood = "idle";
	}

	// Blink
	if (time >= nextBlinkAt && blinkFade <= 0) {
		blinkFade = 1.0;
		nextBlinkAt = time + 2.5 + Math.random() * 3.5;
	}
	if (blinkFade > 0) blinkFade = Math.max(0, blinkFade - dt * 8.0);

	const breath = Math.sin(time * 4.0);
	const bSmall = breath * 0.3; // Scaled breathing for widget-space

	// Default targets in native widget coordinates
	const targets: Blob[] = [
		{ x: CX, y: CY + 1 + bSmall, rx: 8 + bSmall * 0.2, ry: 5 - bSmall * 0.2, rot: 0 },
		{ x: CX - 5, y: CY - 4 + bSmall, rx: 2, ry: 4, rot: -0.4 - bSmall * 0.05 },
		{ x: CX + 5, y: CY - 4 + bSmall, rx: 2, ry: 4, rot: 0.4 + bSmall * 0.05 },
	];

	if (mood === "talking") {
		const bounce = Math.sin(time * 14.0) * 0.15;
		targets[0].y += bounce;
		targets[0].ry -= bounce * 0.3;
	} else if (mood === "excited") {
		const bounce = Math.sin(time * 20.0) * 0.25;
		targets[0].y = CY + 1.5 + bounce;
		targets[0].ry = 4.5 - bounce * 0.5;
		targets[0].rx = 9 + bounce * 0.5;
		targets[1].rot = -0.8;
		targets[1].y += 1.5;
		targets[2].rot = 0.8;
		targets[2].y += 1.5;
	}

	// Spring physics
	const K = 150.0;
	const D = 12.0;
	for (let i = 0; i < 3; i++) {
		for (const prop of ["x", "y", "rx", "ry", "rot"] as const) {
			const force = (targets[i][prop] - blobs[i][prop]) * K;
			vels[i][prop] += force * dt;
			vels[i][prop] -= vels[i][prop] * D * dt;
			blobs[i][prop] += vels[i][prop] * dt;
		}
	}
}

function renderFrame(audioLevel: number) {
	outColors.fill(0);

	const bx = blobs[0].x;
	const by = blobs[0].y;
	const f = flavor;
	const eyeOpen = Math.max(0.1, 1.0 - blinkFade);
	const eyeScaleY = eyeOpen;

	for (let y = 0; y < LOGICAL_H; y++) {
		for (let x = 0; x < SW; x++) {
			// Native widget coordinates — no mapping needed
			const d = getDensity(x, y);
			const idx = (y * SW + x) * 3;

			if (d > 1.0) {
				// Normal + lighting
				const eps = 0.3;
				const dx = getDensity(x + eps, y) - getDensity(x - eps, y);
				const dy = getDensity(x, y + eps) - getDensity(x, y - eps);
				let nx = -dx * 18.0;
				let ny = -dy * 18.0;
				let nz = 1.0;
				const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
				nx /= len; ny /= len; nz /= len;

				const diffuse = Math.max(0, nx * 0.4 + ny * -0.7 + nz * 0.6);
				let hx = 0.4, hy = -0.7, hz = 1.6;
				const hlen = Math.sqrt(hx * hx + hy * hy + hz * hz);
				hx /= hlen; hy /= hlen; hz /= hlen;
				const spec = Math.pow(Math.max(0, nx * hx + ny * hy + nz * hz), 40.0);
				const rim = Math.pow(1.0 - Math.max(0, nz), 4.0);

				const amb = 0.4;
				let r = f.r * (diffuse * 0.6 + amb) + spec * 255 + rim * 50;
				let g = f.g * (diffuse * 0.6 + amb) + spec * 255 + rim * 30;
				let b = f.b * (diffuse * 0.6 + amb) + spec * 255 + rim * 100;

				let isFace = false;

				// Eyes — ±3 from body center, 1 above center
				const ex1 = x - (bx - 3), ey1 = y - (by - 1);
				const ex2 = x - (bx + 3), ey2 = y - (by - 1);

				if (ex1 * ex1 + (ey1 * ey1) / (eyeScaleY * eyeScaleY) < 2.5) {
					if (ex1 * ex1 + (ey1 + 0.5) * (ey1 + 0.5) < 0.8 && eyeOpen > 0.5) {
						r = 255; g = 255; b = 255; // catchlight
					} else { r = 40; g = 30; b = 40; } // pupil
					isFace = true;
				}
				if (ex2 * ex2 + (ey2 * ey2) / (eyeScaleY * eyeScaleY) < 2.5) {
					if (ex2 * ex2 + (ey2 + 0.5) * (ey2 + 0.5) < 0.8 && eyeOpen > 0.5) {
						r = 255; g = 255; b = 255;
					} else { r = 40; g = 30; b = 40; }
					isFace = true;
				}

				// Blush — ±4.5 from center, 1 below center
				if (!isFace) {
					const blx1 = x - (bx - 4.5), bly1 = y - (by + 1);
					const blx2 = x - (bx + 4.5), bly2 = y - (by + 1);
					if (blx1 * blx1 + bly1 * bly1 < 1.2 || blx2 * blx2 + bly2 * bly2 < 1.2) {
						r = f.blushR; g = f.blushG; b = f.blushB;
						isFace = true;
					}
				}

				// Mouth — 1.5 below body center
				if (!isFace) {
					const mx = x - bx, my = y - (by + 1.5);
					if (talkTimer > 0) {
						const talkPhase = Math.floor(time * 12) % 3;
						if (talkPhase === 0 && (((mx + 0.6) * (mx + 0.6) + my * my < 0.6 && my > 0) || ((mx - 0.6) * (mx - 0.6) + my * my < 0.6 && my > 0))) {
							r = f.mouthR; g = f.mouthG; b = f.mouthB; // :3
						} else if (talkPhase === 1 && mx * mx + my * my < 0.4) {
							r = f.mouthR; g = f.mouthG; b = f.mouthB; // small o
						} else if (talkPhase === 2 && mx * mx * 0.6 + my * my < 0.9) {
							r = f.mouthR; g = f.mouthG; b = f.mouthB; // big O
						}
					} else {
						// Cute ':3' mouth
						const dx1 = mx + 0.6, dy1 = my;
						const dx2 = mx - 0.6, dy2 = my;
						if ((dx1 * dx1 + dy1 * dy1 < 0.6 && dy1 > 0) || (dx2 * dx2 + dy2 * dy2 < 0.6 && dy2 > 0)) {
							r = f.mouthR; g = f.mouthG; b = f.mouthB;
						}
					}
				}

				outColors[idx] = Math.min(255, Math.max(0, r));
				outColors[idx + 1] = Math.min(255, Math.max(0, g));
				outColors[idx + 2] = Math.min(255, Math.max(0, b));
			}
		}
	}
}

/**
 * Render the slime to an array of ANSI-colored string lines.
 *
 * @param audioLevel - 0.0 to 1.0, drives mood and mouth animation
 * @param dt - time delta in seconds since last frame
 * @returns string[] of SH lines, each SW characters wide with ANSI color codes
 */
export function renderSlime(audioLevel: number, dt: number): string[] {
	updatePhysics(dt, audioLevel);
	renderFrame(audioLevel);

	const lines: string[] = [];
	for (let cy = 0; cy < SH; cy++) {
		let line = "";
		let lastFg = "";
		let lastBg = "";

		for (let cx = 0; cx < SW; cx++) {
			const tIdx = (cy * 2 * SW + cx) * 3;
			const bIdx = ((cy * 2 + 1) * SW + cx) * 3;

			const tr = outColors[tIdx], tg = outColors[tIdx + 1], tb = outColors[tIdx + 2];
			const br = outColors[bIdx], bg = outColors[bIdx + 1], bb = outColors[bIdx + 2];

			const tTrans = tr === 0 && tg === 0 && tb === 0;
			const bTrans = br === 0 && bg === 0 && bb === 0;

			if (tTrans && bTrans) {
				if (lastFg || lastBg) {
					line += "\x1b[39m\x1b[49m";
					lastFg = ""; lastBg = "";
				}
				line += " ";
			} else if (bTrans) {
				const fg = `\x1b[38;2;${tr};${tg};${tb}m`;
				if (lastFg !== fg) { line += fg; lastFg = fg; }
				if (lastBg) { line += "\x1b[49m"; lastBg = ""; }
				line += "▀";
			} else if (tTrans) {
				const fg = `\x1b[38;2;${br};${bg};${bb}m`;
				if (lastFg !== fg) { line += fg; lastFg = fg; }
				if (lastBg) { line += "\x1b[49m"; lastBg = ""; }
				line += "▄";
			} else {
				const fg = `\x1b[38;2;${tr};${tg};${tb}m`;
				const bgStr = `\x1b[48;2;${br};${bg};${bb}m`;
				if (lastFg !== fg) { line += fg; lastFg = fg; }
				if (lastBg !== bgStr) { line += bgStr; lastBg = bgStr; }
				line += "▀";
			}
		}
		line += "\x1b[0m";
		lines.push(line);
	}
	return lines;
}

/** Reset slime state (e.g., when recording stops) */
export function resetSlime() {
	time = 0;
	mood = "idle";
	blinkFade = 0;
	talkTimer = 0;
	// Reset blobs to native widget-space defaults
	blobs[0] = { x: CX, y: CY + 1, rx: 8, ry: 5, rot: 0 };
	blobs[1] = { x: CX - 5, y: CY - 4, rx: 2, ry: 4, rot: -0.4 };
	blobs[2] = { x: CX + 5, y: CY - 4, rx: 2, ry: 4, rot: 0.4 };
	for (const v of vels) { v.x = 0; v.y = 0; v.rx = 0; v.ry = 0; v.rot = 0; }
}

/** Width of the slime in characters (for layout) */
export const SLIME_WIDTH = SW;
/** Height of the slime in character rows (for layout) */
export const SLIME_HEIGHT = SH;
