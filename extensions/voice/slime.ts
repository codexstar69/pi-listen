/**
 * Gelatinous slime pet — adapted for Pi voice widget.
 *
 * Renders a soft-body metaball slime creature using half-block characters
 * and 24-bit ANSI color. Driven by audioLevel instead of keyboard input.
 *
 * Based on /Users/codex/gelatinous_slime_pet.ts
 */

// --- Configuration ---
const SW = 40;   // Widget pixel width (characters)
const SH = 10;   // Widget pixel height (character rows, each = 2 logical pixels)
const LOGICAL_H = SH * 2; // 20 logical pixels tall

// Scale factor from original 80×72 coordinate space
const SCALE_X = SW / 80;
const SCALE_Y = LOGICAL_H / 72;

// --- Slime State ---
let time = 0;
let mood: "idle" | "talking" | "sleeping" | "excited" = "idle";
let blinkFade = 0;
let nextBlinkAt = 2 + Math.random() * 3;
let talkTimer = 0;

// Flavor (Catppuccin Mocha - Mauve)
const flavor = { r: 203, g: 166, b: 247, blushR: 245, blushG: 130, blushB: 186, mouthR: 180, mouthG: 120, mouthB: 200 };

// Metablobs: 0: Body, 1: Left Ear, 2: Right Ear
interface Blob { x: number; y: number; rx: number; ry: number; rot: number; }
const blobs: Blob[] = [
	{ x: 40, y: 40, rx: 20, ry: 16, rot: 0 },
	{ x: 26, y: 22, rx: 5, ry: 12, rot: -0.4 },
	{ x: 54, y: 22, rx: 5, ry: 12, rot: 0.4 },
];

const vels: Blob[] = Array.from({ length: 3 }, () => ({ x: 0, y: 0, rx: 0, ry: 0, rot: 0 }));

// Output buffer
const outColors = new Uint8Array(SW * LOGICAL_H * 3);

// --- Math ---
function getDensity(x: number, y: number): number {
	let d = 0;
	for (const b of blobs) {
		let dx = x - b.x;
		let dy = y - b.y;
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

	// Default targets
	const targets: Blob[] = [
		{ x: 40, y: 40 + breath * 1.5, rx: 20 + breath * 0.5, ry: 16 - breath * 0.5, rot: 0 },
		{ x: 26, y: 22 + breath * 1.5, rx: 5, ry: 12, rot: -0.4 - breath * 0.05 },
		{ x: 54, y: 22 + breath * 1.5, rx: 5, ry: 12, rot: 0.4 + breath * 0.05 },
	];

	if (mood === "talking") {
		const bounce = Math.sin(time * 14.0) * 0.3;
		targets[0].y += bounce * 2;
		targets[0].ry -= bounce;
	} else if (mood === "excited") {
		const bounce = Math.sin(time * 20.0) * 0.5;
		targets[0].y = 42 + bounce * 3;
		targets[0].ry = 15 - bounce * 1.5;
		targets[0].rx = 22 + bounce * 1.5;
		targets[1].rot = -0.8;
		targets[1].y += 4;
		targets[2].rot = 0.8;
		targets[2].y += 4;
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
			// Map widget coords back to 80×72 space
			const ox = x / SCALE_X;
			const oy = y / SCALE_Y;

			const d = getDensity(ox, oy);
			const idx = (y * SW + x) * 3;

			if (d > 1.0) {
				// Normal + lighting
				const eps = 0.5;
				const dx = getDensity(ox + eps, oy) - getDensity(ox - eps, oy);
				const dy = getDensity(ox, oy + eps) - getDensity(ox, oy - eps);
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

				// Eyes
				const ex1 = ox - (bx - 8), ey1 = oy - (by - 2);
				const ex2 = ox - (bx + 8), ey2 = oy - (by - 2);

				if (ex1 * ex1 + (ey1 * ey1) / (eyeScaleY * eyeScaleY) < 14) {
					if (ex1 * ex1 + (ey1 + 1.5) * (ey1 + 1.5) < 3.5 && eyeOpen > 0.5) {
						r = 255; g = 255; b = 255;
					} else if ((ex1 + 1.5) * (ex1 + 1.5) + (ey1 - 1.5) * (ey1 - 1.5) < 1.0 && eyeOpen > 0.5) {
						r = 255; g = 255; b = 255;
					} else { r = 40; g = 30; b = 40; }
					isFace = true;
				}
				if (ex2 * ex2 + (ey2 * ey2) / (eyeScaleY * eyeScaleY) < 14) {
					if (ex2 * ex2 + (ey2 + 1.5) * (ey2 + 1.5) < 3.5 && eyeOpen > 0.5) {
						r = 255; g = 255; b = 255;
					} else if ((ex2 + 1.5) * (ex2 + 1.5) + (ey2 - 1.5) * (ey2 - 1.5) < 1.0 && eyeOpen > 0.5) {
						r = 255; g = 255; b = 255;
					} else { r = 40; g = 30; b = 40; }
					isFace = true;
				}

				// Blush
				if (!isFace) {
					const blx1 = ox - (bx - 12), bly1 = oy - (by + 2);
					const blx2 = ox - (bx + 12), bly2 = oy - (by + 2);
					if (blx1 * blx1 + bly1 * bly1 < 6 || blx2 * blx2 + bly2 * bly2 < 6) {
						r = f.blushR; g = f.blushG; b = f.blushB;
						isFace = true;
					}
				}

				// Mouth
				if (!isFace) {
					const mx = ox - bx, my = oy - (by + 4);
					if (talkTimer > 0) {
						const talkPhase = Math.floor(time * 12) % 3;
						if (talkPhase === 0 && (((mx + 1.5) * (mx + 1.5) + (my - 1) * (my - 1) < 3.0 && my > 1) || ((mx - 1.5) * (mx - 1.5) + (my - 1) * (my - 1) < 3.0 && my > 1))) {
							r = f.mouthR; g = f.mouthG; b = f.mouthB;
						} else if (talkPhase === 1 && mx * mx + (my - 1) * (my - 1) < 2) {
							r = f.mouthR; g = f.mouthG; b = f.mouthB;
						} else if (talkPhase === 2 && mx * mx * 0.6 + (my - 1) * (my - 1) < 5) {
							r = f.mouthR; g = f.mouthG; b = f.mouthB;
						}
					} else {
						const dx1 = mx + 1.5, dy1 = my - 1.0;
						const dx2 = mx - 1.5, dy2 = my - 1.0;
						if ((dx1 * dx1 + dy1 * dy1 < 3.0 && dy1 > 0) || (dx2 * dx2 + dy2 * dy2 < 3.0 && dy2 > 0)) {
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
	// Reset blobs to defaults
	blobs[0] = { x: 40, y: 40, rx: 20, ry: 16, rot: 0 };
	blobs[1] = { x: 26, y: 22, rx: 5, ry: 12, rot: -0.4 };
	blobs[2] = { x: 54, y: 22, rx: 5, ry: 12, rot: 0.4 };
	for (const v of vels) { v.x = 0; v.y = 0; v.rx = 0; v.ry = 0; v.rot = 0; }
}
