/**
 * Lumo Companion — adapted for Pi voice widget.
 *
 * A full 3D raymarched creature with physics, particles, speech bubbles,
 * moods, and interactive commands. Driven by audio level for mouth animation.
 *
 * Based on /Users/codex/lumo_companion.ts
 */

// ─── Rendering Config ────────────────────────────────────────────────────────
// Widget dimensions — set once, used by renderLumo
let W = 50;
let H = 14; // character rows (each = 2 logical pixels via half-block)

const PHYSICS_DT = 0.016; // 60fps physics sub-stepping

// ─── Pet State ───────────────────────────────────────────────────────────────
type State = "idle" | "walk" | "flip" | "sleep" | "excited" | "chasing" | "fetching" | "singing" | "offscreen" | "peek";
let currentState: State = "idle";

let time = 0;
let blinkFade = 0;
let actionTimer = 0;
let speechTimer = 0;
let speechText = "";

// Needs
let hunger = 100;
let energy = 100;
let lastNeedsTick = 0;

// Themes
const themes = [
	{ name: "Cloud", r: 245, g: 250, b: 255 },
	{ name: "Cotton Candy", r: 255, g: 210, b: 230 },
	{ name: "Mint Drop", r: 200, g: 255, b: 220 },
	{ name: "Sunset Gold", r: 255, g: 225, b: 180 },
];
let activeTheme = 0;

// Physical position
let posX = 0, posY = 0, posZ = 0;
let lookX = 0, lookY = 0;
let isWalking = false, isFlipping = false, flipPhase = 0;
let targetX = 0;
let bounceY = 0;
let isSleeping = false;
let breathe = 0;

// Audio-driven talking
let isTalking = false;
let talkAudioLevel = 0;

// Interactables
let ffX = 0, ffY = 0, ffZ = 0;
interface Food { x: number; y: number; vy: number; }
const foods: Food[] = [];
let ballX = -10, ballY = -10, ballZ = 0, ballVx = 0, ballVy = 0, ballVz = 0, hasBall = false;

interface Particle {
	x: number; y: number; vx: number; vy: number;
	char: string; r: number; g: number; b: number; life: number; type: string;
}
const particles: Particle[] = [];

let screenChars: string[][] = [];
let screenColors: string[][] = [];

function allocBuffers() {
	screenChars = Array.from({ length: H }, () => Array(W).fill(" "));
	screenColors = Array.from({ length: H }, () => Array(W).fill(""));
}
allocBuffers();

interface RenderObj {
	id: string; mat: number;
	x: number; y: number; z: number;
	r?: number; rx?: number; ry?: number; rot?: number;
	s?: number; c?: number;
}

function say(text: string, duration = 4.0) {
	speechText = text;
	speechTimer = duration;
}

function project2D(x: number, y: number): [number, number] {
	const effectDim = Math.max(50, Math.min(W, H * 2.0));
	const scale = 2.0 / effectDim;
	const cx = (x / scale) + (W / 2.0);
	const cy = (y / scale + H) / 2.0;
	return [Math.floor(cx), Math.floor(cy)];
}

function drawSpeechBubble(text: string, bx: number, by: number) {
	if (text.length > W - 10) text = text.substring(0, W - 13) + "...";
	const pad = 2, width = text.length + pad * 2;
	const startX = Math.floor(bx - width / 2), startY = Math.floor(by - 3);
	if (startY < 0 || startY >= H) return;

	const top = "╭" + "─".repeat(width - 2) + "╮";
	const mid = "│ " + text + " │";
	let tailPos = Math.floor(width / 2);
	if (startX < 0) tailPos = 2;
	if (startX + width > W) tailPos = width - 3;
	const bot = "╰" + "─".repeat(Math.max(0, tailPos - 1)) + "v" + "─".repeat(Math.max(0, width - tailPos - 2)) + "╯";

	const drawLine = (ly: number, str: string) => {
		if (ly >= 0 && ly < H) {
			for (let i = 0; i < str.length; i++) {
				const lx = startX + i;
				if (lx >= 0 && lx < W) {
					screenChars[ly][lx] = str[i];
					screenColors[ly][lx] = "\x1b[38;5;234m\x1b[48;5;255m";
				}
			}
		}
	};
	drawLine(startY, top); drawLine(startY + 1, mid); drawLine(startY + 2, bot);
}

function fbm(x: number, y: number): number {
	return Math.sin(x * 15 + time * 2) * Math.sin(y * 15 + time * 1.5) * 0.04 +
		Math.sin(x * 30 - time) * Math.cos(y * 30) * 0.02;
}

function getWeatherAndTime() {
	const h = new Date().getHours();
	let rTop = 0, gTop = 0, bTop = 0, rBot = 0, gBot = 0, bBot = 0;
	if (h >= 6 && h < 17) { rTop = 40; gTop = 120; bTop = 255; rBot = 180; gBot = 220; bBot = 255; }
	else if (h >= 17 && h < 19) { rTop = 140; gTop = 50; bTop = 120; rBot = 255; gBot = 160; bBot = 100; }
	else { rTop = 10; gTop = 10; bTop = 20; rBot = 25; gBot = 20; bBot = 40; }
	return { rTop, gTop, bTop, rBot, gBot, bBot, isNight: h >= 19 || h < 6 };
}

function getObjHit(px: number, py: number, objects: RenderObj[]) {
	let hitObj: RenderObj | null = null;
	let hitNx = 0, hitNy = 0, hitNz = 1;
	let hitU = 0, hitV = 0;

	for (const obj of objects) {
		let dx = px - obj.x;
		let dy = py - obj.y;
		const maxR = Math.max(obj.rx || obj.r || 1.0, obj.ry || obj.r || 1.0);
		if (Math.abs(dx) > maxR + 0.35 || Math.abs(dy) > maxR + 0.35) continue;

		if (obj.s !== undefined && obj.c !== undefined) {
			const nx = dx * obj.c + dy * obj.s;
			const ny = -dx * obj.s + dy * obj.c;
			dx = nx; dy = ny;
		}

		const rx = obj.rx || obj.r || 1;
		const ry = obj.ry || obj.r || 1;
		let dist = Math.sqrt((dx * dx) / (rx * rx) + (dy * dy) / (ry * ry));

		let fluff = 0;
		if (obj.id === "body") {
			fluff = fbm(dx, dy);
			const faceDist = Math.sqrt(dx * dx + dy * dy);
			const faceMask = Math.max(0, 1.0 - faceDist * 4.0);
			fluff *= (1.0 - faceMask);
			if (isSleeping) fluff *= 0.3;
		} else if (obj.id === "tail") {
			fluff = Math.sin(Math.atan2(dy, dx) * 5 + time * 3) * 0.2;
		} else if (obj.id === "pillow") {
			fluff = Math.sin(dx * 5) * Math.cos(dy * 10) * 0.1;
		}

		if (dist < 1.0 + fluff) {
			hitObj = obj;
			hitNx = dx / rx; hitNy = dy / ry;
			const nlen = Math.sqrt(hitNx * hitNx + hitNy * hitNy);
			if (nlen > 1.0) { hitNx /= nlen; hitNy /= nlen; }
			hitNz = Math.sqrt(Math.max(0, 1.0 - hitNx * hitNx - hitNy * hitNy));
			hitU = hitNx; hitV = hitNy;
			if (obj.s !== undefined && obj.c !== undefined) {
				const nnx = hitNx * obj.c - hitNy * obj.s;
				const nny = hitNx * obj.s + hitNy * obj.c;
				hitNx = nnx; hitNy = nny;
			}
		}
	}
	return { hitObj, hitNx, hitNy, hitNz, hitU, hitV };
}

function shadeObject(hit: ReturnType<typeof getObjHit>, px: number, py: number, objects: RenderObj[]): [number, number, number] {
	const { hitObj, hitNx, hitNy, hitNz, hitU, hitV } = hit;
	if (!hitObj) return [-1, -1, -1];

	let r = 255, g = 255, b = 255, gloss = 0;
	const th = themes[activeTheme];

	if (hitObj.mat === 1) {
		r = th.r; g = th.g; b = th.b;
		if (hitNy > 0.2) {
			const belly = Math.min(1.0, (hitNy - 0.2) * 1.5);
			r = r * (1 - belly) + 255 * belly; g = g * (1 - belly) + 250 * belly; b = b * (1 - belly) + 245 * belly;
		}
		if (hitObj.id === "body" && hitNy < -0.3) {
			const spot = Math.sin(hitNx * 10) * Math.cos(hitNy * 8);
			if (spot > 0.6) { r = Math.max(0, r - 40); g = Math.max(0, g - 20); b = Math.max(0, b - 10); }
		}
		if (hitObj.id === "body") {
			let bdx = px - hitObj.x, bdy = py - hitObj.y;
			if (isFlipping) {
				const s = Math.sin(-flipPhase), c = Math.cos(-flipPhase);
				const nx = bdx * c - bdy * s, ny = bdx * s + bdy * c;
				bdx = nx; bdy = ny;
			}
			// Blush
			const blx1 = bdx + 0.16, bly1 = bdy - 0.04;
			const blx2 = bdx - 0.16, bly2 = bdy - 0.04;
			const blush = Math.exp(-(blx1 * blx1 + bly1 * bly1) * 80) + Math.exp(-(blx2 * blx2 + bly2 * bly2) * 80);
			if (!isSleeping) {
				r = r * (1 - blush) + 255 * blush; g = g * (1 - blush) + 100 * blush; b = b * (1 - blush) + 120 * blush;
			}
			// Eyes
			const isTired = (energy < 20 || hunger < 30) && !isSleeping;
			const eyeOpen = isSleeping ? 0.05 : (isTired ? 0.4 : 1.0) - blinkFade;
			const ex1 = bdx - lookX * 0.08 + 0.1, ey1 = bdy - lookY * 0.05 + 0.02;
			const ex2 = bdx - lookX * 0.08 - 0.1, ey2 = bdy - lookY * 0.05 + 0.02;

			if (isSleeping || currentState === "singing") {
				if (isSleeping) {
					if ((Math.abs(ey1) < 0.01 && Math.abs(ex1) < 0.05) || (Math.abs(ey2) < 0.01 && Math.abs(ex2) < 0.05)) { r = 30; g = 20; b = 30; }
				} else {
					if ((Math.abs(ey1 + ex1 * ex1 * 15) < 0.015 && Math.abs(ex1) < 0.06 && ey1 > -ex1 * ex1 * 15) ||
						(Math.abs(ey2 + ex2 * ex2 * 15) < 0.015 && Math.abs(ex2) < 0.06 && ey2 > -ex2 * ex2 * 15)) { r = 30; g = 20; b = 30; }
				}
			} else {
				const eDist1 = ex1 * ex1 + (ey1 * ey1) / (eyeOpen * eyeOpen + 0.001);
				const eDist2 = ex2 * ex2 + (ey2 * ey2) / (eyeOpen * eyeOpen + 0.001);
				if (eDist1 < 0.0025 || eDist2 < 0.0025) {
					r = 25; g = 20; b = 30;
					if (ey1 > 0 || ey2 > 0) { r = 50; g = 180; b = 100; }
					if ((ex1 + 0.012) ** 2 + (ey1 + 0.012) ** 2 < 0.0003 || (ex2 + 0.012) ** 2 + (ey2 + 0.012) ** 2 < 0.0003) {
						if (!isTired) { r = 255; g = 255; b = 255; }
					} else if ((ex1 - 0.015) ** 2 + (ey1 - 0.015) ** 2 < 0.0001 || (ex2 - 0.015) ** 2 + (ey2 - 0.015) ** 2 < 0.0001) {
						if (!isTired) { r = 200; g = 220; b = 255; }
					}
				}
			}
			// Nose
			const nnx = bdx - lookX * 0.06, nny = bdy - lookY * 0.05 - 0.02;
			if (nnx * nnx * 1.5 + nny * nny < 0.0003 && !isSleeping) { r = 30; g = 20; b = 30; }
			// Mouth
			if (!isSleeping && !hasBall) {
				const mx = bdx - lookX * 0.06, my = bdy - lookY * 0.05 - 0.06;
				if ((Math.abs(my - (mx - 0.025) ** 2 * 20 + 0.01) < 0.006 && mx > 0 && mx < 0.05) ||
					(Math.abs(my - (mx + 0.025) ** 2 * 20 + 0.01) < 0.006 && mx < 0 && mx > -0.05)) {
					r = 50; g = 30; b = 40;
				}
				if (currentState === "excited" || currentState === "singing" || speechTimer > 0 || isTalking) {
					const mouthOpen = (speechTimer > 0 || currentState === "singing" || isTalking)
						? (isTalking ? talkAudioLevel * 0.04 + 0.005 : Math.abs(Math.sin(time * 12)) * 0.025)
						: 0.015;
					if (mx * mx + (my + 0.01) ** 2 < mouthOpen && my < -0.01) {
						r = 240; g = 80; b = 100;
						if (my < -0.025) { r = 255; g = 120; b = 140; }
					}
				}
			}
		} else {
			if (hitObj.id === "earL" || hitObj.id === "earR") {
				if (hitU > -0.3 && hitU < 0.3 && hitV > -0.5 && hitV < 0.5) { r = 255; g = 160; b = 180; }
			}
		}
	} else if (hitObj.mat === 2) {
		r = Math.max(0, th.r - 20); g = Math.max(0, th.g - 15); b = Math.max(0, th.b - 10);
		if (hitNy > 0.5) { r = 255; g = 180; b = 190; }
	} else if (hitObj.mat === 3) {
		r = 255; g = 230; b = 90; gloss = 128;
	} else if (hitObj.mat === 5) {
		return [100, 255, 200];
	} else if (hitObj.mat === 6) {
		r = 240; g = 220; b = 180; gloss = 16;
	} else if (hitObj.mat === 7) {
		r = 120; g = 130; b = 140;
	} else if (hitObj.mat === 8) {
		const pulse = Math.sin(time * 6) * 0.5 + 0.5;
		return [255, Math.floor(100 + pulse * 150), Math.floor(150 + pulse * 105)];
	} else if (hitObj.mat === 9) {
		r = 255; g = 60; b = 80;
		const curve = Math.abs(hitNx * 0.7 - hitNy * 0.7);
		if (curve > 0.4 && curve < 0.55) { r = 255; g = 200; b = 200; }
		gloss = 128;
	} else if (hitObj.mat === 10) {
		r = 230; g = 210; b = 220;
		const check = Math.sin(hitU * 20) * Math.sin(hitV * 20);
		if (check > 0) { r = 200; g = 180; b = 200; }
	}

	// Lighting
	let lx = 0.6, ly = -0.7, lz = 0.8;
	const ll = Math.sqrt(lx * lx + ly * ly + lz * lz);
	lx /= ll; ly /= ll; lz /= ll;

	const diff = Math.max(0, hitNx * lx + hitNy * ly + hitNz * lz);
	const wrap = Math.max(0, hitNx * lx + hitNy * ly + hitNz * lz + 0.5) / 1.5;
	const amb = 0.45;

	let ao = 1.0;
	if (hitObj.id === "earL" || hitObj.id === "earR") ao = 0.8;
	if (hitObj.id === "pawL" || hitObj.id === "pawR") ao = 0.7;
	if (hitObj.id === "body" && hitNy > 0.5) ao = 0.6;
	if (hitObj.id === "pillow" && isSleeping) {
		const bodyDist = Math.sqrt((px - posX) ** 2 + (py - posY) ** 2);
		if (bodyDist < 0.4) ao = 0.5 + (bodyDist / 0.4) * 0.5;
	}

	// Firefly light
	const fdx = ffX - px, fdy = ffY - py, fdz = ffZ - (hitObj.z || 0);
	const fDistSq = fdx * fdx + fdy * fdy + fdz * fdz;
	const fll = Math.max(0.001, Math.sqrt(fDistSq));
	const fnx = fdx / fll, fny = fdy / fll, fnz = fdz / fll;
	const fdiff = Math.max(0, hitNx * fnx + hitNy * fny + hitNz * fnz);
	const fatten = 1.0 / (1.0 + fDistSq * 20.0);

	let lightR = (diff * 0.5 + wrap * 0.3 + amb) * ao + fdiff * fatten * 2.0;
	let lightG = (diff * 0.5 + wrap * 0.3 + amb) * ao + fdiff * fatten * 3.0;
	let lightB = (diff * 0.5 + wrap * 0.3 + amb) * ao + fdiff * fatten * 2.5;

	// Antenna glow
	if (hitObj.id === "body") {
		const antObj = objects.find(o => o.id === "antenna_bulb");
		if (antObj) {
			const antDx = px - antObj.x, antDy = py - antObj.y;
			const antDist = Math.sqrt(antDx * antDx + antDy * antDy);
			const antAtten = 1.0 / (1.0 + antDist * antDist * 40.0);
			lightR += antAtten * 1.5; lightG += antAtten * 0.5; lightB += antAtten * 0.8;
		}
	}

	r = Math.min(255, Math.floor(r * lightR));
	g = Math.min(255, Math.floor(g * lightG));
	b = Math.min(255, Math.floor(b * lightB));

	if (gloss > 0 && diff > 0) {
		const spec = Math.pow(Math.max(0, hitNx * lx + hitNy * ly + hitNz * lz), gloss);
		r = Math.min(255, Math.floor(r + spec * 255));
		g = Math.min(255, Math.floor(g + spec * 255));
		b = Math.min(255, Math.floor(b + spec * 255));
	}

	return [r, g, b];
}

function getPixel(px: number, py: number, objects: RenderObj[], skyColors: ReturnType<typeof getWeatherAndTime>): [number, number, number] {
	if (py > 0.6) {
		let shadowDist = Math.sqrt((px - posX) ** 2 + ((py - 0.6) * 2.5) ** 2);
		let shadow = Math.max(0.2, Math.min(1.0, shadowDist / 0.7));
		if (isSleeping) {
			const pillowDist = Math.sqrt(px ** 2 + ((py - 0.6) * 2.5) ** 2);
			shadow = Math.min(shadow, Math.max(0.3, Math.min(1.0, pillowDist / 1.5)));
		}
		if (ballY > 0.4 && ballX !== -10 && !hasBall) {
			const bShadowDist = Math.sqrt((px - ballX) ** 2 + ((py - 0.6) * 2.5) ** 2);
			shadow = Math.min(shadow, Math.max(0.4, Math.min(1.0, bShadowDist / 0.2)));
		}
		const isWood = (Math.sin(px * 10) + Math.sin(py * 40)) > 0;
		const wr = isWood ? 55 : 45, wg = isWood ? 35 : 30, wb = isWood ? 25 : 20;
		const grad = (py - 0.6) / 0.4;
		let fr = Math.floor((wr - grad * 10) * shadow);
		let fg = Math.floor((wg - grad * 10) * shadow);
		let fb = Math.floor((wb - grad * 10) * shadow);
		// Floor reflection
		const refPy = 1.2 - py;
		const refHit = getObjHit(px, refPy, objects);
		if (refHit.hitObj) {
			const refC = shadeObject(refHit, px, refPy, objects);
			fr = Math.floor(fr * 0.7 + refC[0] * 0.3);
			fg = Math.floor(fg * 0.7 + refC[1] * 0.3);
			fb = Math.floor(fb * 0.7 + refC[2] * 0.3);
		}
		return [fr, fg, fb];
	}

	const directHit = getObjHit(px, py, objects);
	if (directHit.hitObj) return shadeObject(directHit, px, py, objects);

	const grad = Math.max(0, (1.0 + py) / 2.0);
	let bgR = Math.floor(skyColors.rTop * (1 - grad) + skyColors.rBot * grad);
	let bgG = Math.floor(skyColors.gTop * (1 - grad) + skyColors.gBot * grad);
	let bgB = Math.floor(skyColors.bTop * (1 - grad) + skyColors.bBot * grad);
	if (skyColors.isNight) {
		const star = Math.sin(px * 80) * Math.cos(py * 80);
		if (star > 0.99) { bgR = 255; bgG = 255; bgB = 255; }
		else if (star > 0.97) { bgR += 50; bgG += 50; bgB += 50; }
	}
	return [bgR, bgG, bgB];
}

function buildObjects(): RenderObj[] {
	breathe = Math.sin(time * (isSleeping ? 1.5 : 3)) * 0.015;
	let earWave = Math.sin(time * 4) * 0.08;
	if (currentState === "excited" || currentState === "fetching" || currentState === "singing") earWave = Math.sin(time * 15) * 0.2;
	if (isWalking) earWave += Math.sin(time * 10) * 0.1;
	const pawSwing = (isWalking || currentState === "chasing" || currentState === "fetching" || currentState === "peek") ? Math.sin(time * 12) * 0.08 : 0;
	const antRot = Math.sin(time * 2.5) * 0.15 + (isWalking || currentState === "fetching" ? Math.sin(time * 12) * 0.3 : 0);

	const objects: RenderObj[] = [];
	if (isSleeping) objects.push({ id: "pillow", mat: 10, x: 0, y: 0.65, rx: 0.6, ry: 0.15, z: posZ - 0.1 });

	objects.push(
		{ id: "antenna_stalk", mat: 7, x: posX + Math.sin(antRot) * 0.08, y: posY + bounceY + breathe - 0.35, rx: 0.012, ry: 0.08, rot: antRot, z: 0.05 },
		{ id: "antenna_bulb", mat: 8, x: posX + Math.sin(antRot) * 0.16, y: posY + bounceY + breathe - 0.42, r: 0.035, z: 0.08 },
		{ id: "body", mat: 1, x: posX, y: posY + bounceY + breathe, r: 0.32, z: 0 },
		{ id: "earL", mat: 1, x: posX - 0.28, y: posY + bounceY + breathe - 0.05, rx: 0.08, ry: 0.22, rot: 0.5 + earWave, z: 0.1 },
		{ id: "earR", mat: 1, x: posX + 0.28, y: posY + bounceY + breathe - 0.05, rx: 0.08, ry: 0.22, rot: -0.5 - earWave, z: 0.1 },
		{ id: "pawL", mat: 2, x: posX - 0.14, y: posY + bounceY + breathe + 0.22, r: 0.05, z: 0.2 + pawSwing },
		{ id: "pawR", mat: 2, x: posX + 0.14, y: posY + bounceY + breathe + 0.22, r: 0.05, z: 0.2 - pawSwing },
		{ id: "tail", mat: 3, x: posX + Math.cos(time * 2) * 0.35, y: posY + bounceY + breathe - 0.05, r: 0.06, z: Math.sin(time * 2) * 0.4 },
		{ id: "firefly", mat: 5, x: ffX, y: ffY, r: 0.015, z: ffZ },
	);

	if (isSleeping) {
		const eL = objects.find(o => o.id === "earL")!;
		const eR = objects.find(o => o.id === "earR")!;
		eL.rot = 1.3; eL.y += 0.08; eL.x -= 0.08;
		eR.rot = -1.3; eR.y += 0.08; eR.x += 0.08;
		const pL = objects.find(o => o.id === "pawL")!;
		const pR = objects.find(o => o.id === "pawR")!;
		pL.y += 0.05; pL.x -= 0.1; pR.y += 0.05; pR.x += 0.1;
	}

	if (ballY !== -10) {
		if (hasBall) objects.push({ id: "ball", mat: 9, x: posX + lookX * 0.05, y: posY + bounceY + 0.05, r: 0.035, z: posZ + 0.15 });
		else objects.push({ id: "ball", mat: 9, x: ballX, y: ballY, r: 0.035, z: 0.15 });
	}

	for (const f of foods) objects.push({ id: "food", mat: 6, x: f.x, y: f.y, r: 0.03, z: 0.1 });

	objects.sort((a, b) => a.z - b.z);
	for (const obj of objects) {
		if (obj.rot !== undefined) { obj.s = Math.sin(obj.rot); obj.c = Math.cos(obj.rot); }
	}
	return objects;
}

function updatePhysics(dt: number) {
	if (actionTimer > 0) actionTimer -= dt;
	if (speechTimer > 0) speechTimer -= dt;

	// Needs decay
	const now = Date.now();
	if (now - lastNeedsTick > 1000) {
		lastNeedsTick = now;
		if (!isSleeping) { energy = Math.max(0, energy - 0.5); hunger = Math.max(0, hunger - 0.8); }
		else { energy = Math.min(100, energy + 5.0); hunger = Math.max(0, hunger - 0.2); }
	}

	// Firefly
	ffX = posX + Math.sin(time * 1.2) * 0.7;
	ffY = Math.sin(time * 2.0) * 0.3 + 0.1;
	ffZ = posZ + Math.sin(time * 0.9) * 0.4;

	// Weather particles
	const isRaining = new Date().getMinutes() % 10 < 3 && !getWeatherAndTime().isNight;
	if (isRaining && Math.random() < 0.3) {
		const effectDim = Math.max(50, Math.min(W, H * 2.0));
		const scale = 2.0 / effectDim;
		particles.push({
			x: (Math.random() - 0.5) * W * scale, y: -H * scale,
			vx: 0.1, vy: 2.0 + Math.random(), char: "|", r: 150, g: 200, b: 255, life: 1.0, type: "rain",
		});
	}

	// Ball physics
	if (ballY !== -10 && !hasBall) {
		ballVy += dt * 5.0;
		ballX += ballVx * dt; ballY += ballVy * dt;
		if (ballY > 0.55) { ballY = 0.55; ballVy *= -0.7; ballVx *= 0.8; if (Math.abs(ballVy) < 0.2) ballVy = 0; if (Math.abs(ballVx) < 0.1) ballVx = 0; }
		if (ballX < -1.5) { ballX = -1.5; ballVx *= -0.8; }
		if (ballX > 1.5) { ballX = 1.5; ballVx *= -0.8; }
	}

	// State machine
	if (currentState === "idle") {
		if (Math.random() < 0.01) blinkFade = 1.0;
		else blinkFade = Math.max(0, blinkFade - dt * 6.0);
		bounceY += (0 - bounceY) * dt * 5.0;
		lookX += (0 - lookX) * dt * 3.0;
		if (ballY !== -10 && !hasBall) { currentState = "fetching"; say("Ball!! 🎾", 2.0); }
		else if (Math.random() < 0.005) {
			if (Math.random() < 0.15) targetX = (Math.random() > 0.5 ? 1 : -1) * 2.5; // occasional offscreen walk
			else targetX = (Math.random() - 0.5) * 1.0;
			currentState = "walk"; isWalking = true;
		}
		else if (Math.random() < 0.003) { currentState = "flip"; isFlipping = true; flipPhase = 0; say("Wheee! 💫"); }
		else if (Math.random() < 0.002) { currentState = "chasing"; actionTimer = 3.0; }
	}
	if (currentState === "walk") {
		const dir = Math.sign(targetX - posX);
		posX += dir * dt * 0.6;
		bounceY = -Math.abs(Math.sin(time * 10)) * 0.08;
		lookX = dir * 0.5;
		if (Math.abs(posX - targetX) < 0.05) {
			isWalking = false; posX = targetX; bounceY = 0; lookX = 0;
			if (Math.abs(posX) >= 2.0) { currentState = "offscreen"; actionTimer = 2.0 + Math.random() * 3.0; }
			else currentState = "idle";
		}
	}
	if (currentState === "offscreen") {
		if (actionTimer <= 0) { currentState = "peek"; actionTimer = 4.0; targetX = Math.sign(posX) * 1.35; isWalking = true; }
	}
	if (currentState === "peek") {
		const dir = Math.sign(targetX - posX);
		if (Math.abs(posX - targetX) > 0.05) {
			posX += dir * dt * 0.4; bounceY = -Math.abs(Math.sin(time * 6)) * 0.05;
			lookX = -Math.sign(posX) * 0.8;
		} else {
			isWalking = false; posX = targetX; bounceY = 0;
			lookX = -Math.sign(posX) * 0.6 + Math.sin(time * 2) * 0.2;
			if (actionTimer < 3.0 && speechTimer <= 0 && Math.random() < 0.05) say("Peekaboo! 👀", 2.0);
			if (actionTimer <= 0) { currentState = "walk"; targetX = 0; isWalking = true; }
		}
	}
	if (currentState === "chasing") {
		const dir = Math.sign(ffX - posX);
		posX += dir * dt * 0.8;
		bounceY = -Math.abs(Math.sin(time * 12)) * 0.1;
		lookX = dir * 0.6; isWalking = true;
		if (actionTimer <= 0) { currentState = "idle"; isWalking = false; bounceY = 0; }
	}
	if (currentState === "flip") {
		flipPhase += dt * Math.PI * 2.0;
		bounceY = -Math.sin(flipPhase) * 0.6;
		if (flipPhase >= Math.PI * 2) { isFlipping = false; bounceY = 0; currentState = "idle"; }
	}
	if (currentState === "sleep") {
		blinkFade = 1.0;
		const dir = Math.sign(0 - posX);
		if (Math.abs(posX) > 0.05) { posX += dir * dt * 1.5; bounceY = -Math.abs(Math.sin(time * 12)) * 0.1; }
		else { posX = 0; bounceY += (0.4 - bounceY) * dt * 5.0; }
		if (Math.random() < 0.02) {
			particles.push({ x: posX + 0.2, y: posY + bounceY, vx: 0.15, vy: -0.2, char: "z", r: 150, g: 200, b: 255, life: 1.2, type: "z" });
		}
		if (actionTimer <= 0) { currentState = "idle"; isSleeping = false; say("What a nice nap! ✨"); }
	}
	if (currentState === "excited") {
		blinkFade = 1.0;
		bounceY = -Math.abs(Math.sin(time * 12) * 0.15);
		if (Math.random() < 0.15) {
			particles.push({ x: posX + (Math.random() - 0.5) * 0.6, y: posY + bounceY + (Math.random() - 0.5) * 0.4, vx: (Math.random() - 0.5) * 0.4, vy: -0.4 - Math.random() * 0.4, char: "✨", r: 255, g: 255, b: 150, life: 1.0, type: "sparkle" });
		}
		if (actionTimer <= 0) currentState = "idle";
	}
	if (currentState === "singing") {
		blinkFade = 1.0;
		bounceY = -Math.abs(Math.sin(time * 8) * 0.1);
		lookX = Math.sin(time * 4) * 0.3;
		if (Math.random() < 0.08) {
			particles.push({ x: posX + (Math.random() - 0.5) * 0.6, y: posY + bounceY - 0.4, vx: (Math.random() - 0.5) * 0.4, vy: -0.6 - Math.random() * 0.4, char: Math.random() > 0.5 ? "♪" : "♫", r: 255, g: 150, b: 200, life: 1.5, type: "note" });
		}
		if (actionTimer <= 0) currentState = "idle";
	}
	if (currentState === "fetching") {
		if (!hasBall) {
			const dir = Math.sign(ballX - posX);
			posX += dir * dt * 1.5;
			bounceY = -Math.abs(Math.sin(time * 18)) * 0.15;
			lookX = dir * 0.5;
			if (Math.abs(posX - ballX) < 0.15 && Math.abs(posY + bounceY - ballY) < 0.3) { hasBall = true; say("Got it! 🎾"); }
		} else {
			const dir = Math.sign(0 - posX);
			posX += dir * dt * 0.8;
			bounceY = -Math.abs(Math.sin(time * 15)) * 0.1;
			lookX = dir * 0.5;
			if (Math.abs(posX) < 0.08) {
				hasBall = false; ballX = posX + 0.15; ballY = 0.5; ballVx = 0.8; ballVy = -1.5;
				currentState = "excited"; actionTimer = 2.0; say("Here you go! ✨");
			}
		}
	}

	// Food physics & eating
	for (let i = foods.length - 1; i >= 0; i--) {
		const f = foods[i];
		f.vy += dt * 2.0; f.y += f.vy * dt;
		if (f.y >= 0.5) { f.y = 0.5; f.vy = 0; }
		if (Math.sqrt((f.x - posX) ** 2 + (f.y - (posY + bounceY)) ** 2) < 0.25 && !isSleeping) {
			currentState = "excited"; actionTimer = 2.0;
			for (let k = 0; k < 5; k++) {
				particles.push({ x: f.x, y: f.y, vx: (Math.random() - 0.5) * 0.4, vy: -0.2 - Math.random() * 0.3, char: "*", r: 255, g: 255, b: 200, life: 1.0, type: "crumb" });
			}
			hunger = Math.min(100, hunger + 20);
			say("Yum! 🍪", 2.0);
			foods.splice(i, 1);
		}
	}

	// Particles
	for (let i = particles.length - 1; i >= 0; i--) {
		const p = particles[i];
		p.x += p.vx * dt; p.y += p.vy * dt;
		if (p.type === "z") p.x += Math.sin(p.y * 4.0) * 0.005;
		if (p.type === "note") p.x += Math.sin(p.y * 6.0) * 0.01;
		if (p.type === "rain" && p.y > 0.6) { p.type = "splash"; p.char = "."; p.vy = -0.5; p.vx = (Math.random() - 0.5) * 0.5; p.life = 0.2; }
		p.life -= dt * 0.8;
		if (p.life <= 0) particles.splice(i, 1);
	}
}

function renderToBuffers() {
	const effectDim = Math.max(50, Math.min(W, H * 2.0));
	const scale = 2.0 / effectDim;
	const objects = buildObjects();
	const skyColors = getWeatherAndTime();

	for (let cy = 0; cy < H; cy++) {
		for (let cx = 0; cx < W; cx++) {
			const px = (cx - W / 2.0) * scale;
			const py1 = (cy * 2.0 - H) * scale;
			const py2 = (cy * 2.0 + 1.0 - H) * scale;
			const c1 = getPixel(px, py1, objects, skyColors);
			const c2 = getPixel(px, py2, objects, skyColors);
			screenChars[cy][cx] = "▀";
			screenColors[cy][cx] = `\x1b[38;2;${c1[0]};${c1[1]};${c1[2]}m\x1b[48;2;${c2[0]};${c2[1]};${c2[2]}m`;
		}
	}

	// Overlay particles
	for (const p of particles) {
		const [scX, scY] = project2D(p.x, p.y);
		if (scX >= 0 && scX < W && scY >= 0 && scY < H * 2) {
			const realCy = Math.floor(scY / 2);
			if (realCy >= 0 && realCy < H) {
				screenChars[realCy][scX] = p.char;
				const bgMatch = screenColors[realCy][scX].match(/\x1b\[48;2;\d+;\d+;\d+m/);
				const bg = bgMatch ? bgMatch[0] : "\x1b[49m";
				screenColors[realCy][scX] = `\x1b[38;2;${p.r};${p.g};${p.b}m${bg}`;
			}
		}
	}

	// Speech bubble
	if (speechTimer > 0 && speechText !== "") {
		const [scX, scY] = project2D(posX, posY + bounceY - 0.6);
		drawSpeechBubble(speechText, scX, Math.floor(scY / 2));
	}
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Render the Lumo companion to an array of ANSI-colored string lines.
 *
 * @param width - Available widget width in characters
 * @param audioLevel - 0.0 to 1.0, drives mouth animation during recording
 * @param dt - Time delta in seconds since last frame
 * @returns string[] of H lines, each with ANSI color codes
 */
export function renderLumo(width: number, audioLevel: number, dt: number): string[] {
	// Adapt dimensions if width changed
	if (width !== W && width > 10) {
		W = width;
		allocBuffers();
	}

	// Update talking state from audio
	talkAudioLevel = audioLevel;

	// Sub-step physics for stability
	time += dt;
	let remaining = dt;
	while (remaining > 0) {
		const step = Math.min(remaining, PHYSICS_DT);
		remaining -= step;
		updatePhysics(step);
	}

	renderToBuffers();

	// Convert buffers to string lines
	const lines: string[] = [];
	for (let cy = 0; cy < H; cy++) {
		let line = "";
		let lastColor = "";
		for (let cx = 0; cx < W; cx++) {
			if (screenColors[cy][cx] !== lastColor) {
				line += screenColors[cy][cx];
				lastColor = screenColors[cy][cx];
			}
			line += screenChars[cy][cx];
		}
		line += "\x1b[0m";
		lines.push(line);
	}

	// ── Status bar with command hints ──
	const border = "\x1b[38;5;239m";
	const bar = "─".repeat(Math.max(0, W - 2));
	lines.push(`${border}╭${bar}╮\x1b[0m`);

	// Command hints
	let menu = "[w] Wake  [p] Pet  [f] Feed  [b] Ball  [m] Music  [c] Color  [o] Hide  [s] Sleep  [d] Flip";
	if (isTalking) menu = "🎤 Recording...  " + menu;
	if (menu.length > Math.max(0, W - 6)) menu = menu.substring(0, Math.max(0, W - 9)) + "...";
	lines.push(`${border}│\x1b[0m  \x1b[38;5;250m${menu}\x1b[0m`);

	// State message
	let stateMsg = "";
	if (hunger < 30) stateMsg = "Pompom's tummy is rumbling... (Press f to feed)";
	else if (energy < 20 && !isSleeping) stateMsg = "Pompom is looking very sleepy... (Press s to sleep)";
	else if (currentState === "excited") stateMsg = "Pompom is super happy! ✨";
	else if (isSleeping) stateMsg = "Pompom is dreaming on their favorite pillow... zZz";
	else if (currentState === "walk") stateMsg = "Pompom is wandering around... 🐾";
	else if (currentState === "chasing") stateMsg = "Pompom is chasing a firefly! ✨";
	else if (currentState === "fetching") stateMsg = "Pompom is playing fetch! 🎾";
	else if (currentState === "singing") stateMsg = "Pompom is singing a magical melody! 🎵";
	else if (currentState === "peek") stateMsg = "Pompom is sneaking back in... 👀";
	else if (currentState === "offscreen") stateMsg = "Where did Pompom go? 🤔";
	else if (isTalking) stateMsg = "Pompom is listening to you speak... 🎤";
	else stateMsg = "Pompom floats gently in the terminal.";
	if (stateMsg.length > Math.max(0, W - 6)) stateMsg = stateMsg.substring(0, Math.max(0, W - 9)) + "...";
	lines.push(`${border}│\x1b[0m  \x1b[38;5;153m${stateMsg}\x1b[0m`);

	lines.push(`${border}╰${bar}╯\x1b[0m`);

	return lines;
}

/** Set talking state (driven by voice recording) */
export function lumoSetTalking(active: boolean) {
	isTalking = active;
	if (active && currentState !== "excited" && currentState !== "singing") {
		speechTimer = 0; // Clear auto-speech when user is talking
	}
}

/** Handle a user keypress command */
export function lumoKeypress(key: string) {
	if (key === "p") { currentState = "excited"; actionTimer = 2.5; isSleeping = false; say("Purrrrr... ♥"); }
	else if (key === "w") { currentState = "idle"; isSleeping = false; blinkFade = 0; say("I'm awake! 👀"); }
	else if (key === "s") { currentState = "sleep"; isSleeping = true; actionTimer = 10; say("Time for a nap... zZz"); }
	else if (key === "f") {
		isSleeping = false; currentState = "idle";
		foods.push({ x: posX + (Math.random() - 0.5) * 0.4, y: -0.8, vy: 0 });
	}
	else if (key === "b") {
		isSleeping = false;
		if (ballY === 0.55 && !hasBall && Math.abs(posX - ballX) < 0.4) { ballVy = -1.8; ballVx = (Math.random() - 0.5) * 2.5; say("Boing! 🎾", 2.0); }
		else { ballX = posX + (Math.random() > 0.5 ? 0.8 : -0.8); ballY = -0.4; ballVx = (Math.random() - 0.5) * 1.5; ballVy = -1.2; hasBall = false; }
	}
	else if (key === "m") { isSleeping = false; currentState = "singing"; actionTimer = 5.0; say("La la la~ 🎵"); }
	else if (key === "c") { activeTheme = (activeTheme + 1) % themes.length; }
	else if (key === "d") { currentState = "flip"; isFlipping = true; flipPhase = 0; isSleeping = false; }
	else if (key === "o") { isSleeping = false; currentState = "walk"; targetX = (Math.random() > 0.5 ? 1 : -1) * 2.5; isWalking = true; }
}

/** Reset companion state */
export function resetLumo() {
	time = 0; currentState = "idle"; blinkFade = 0; actionTimer = 0;
	speechTimer = 0; speechText = "";
	posX = 0; posY = 0; posZ = 0; bounceY = 0; lookX = 0; lookY = 0;
	isWalking = false; isFlipping = false; isSleeping = false; isTalking = false;
	talkAudioLevel = 0; flipPhase = 0;
	hunger = 100; energy = 100; lastNeedsTick = 0;
	activeTheme = 0;
	ballX = -10; ballY = -10; ballVx = 0; ballVy = 0; ballVz = 0; hasBall = false;
	ffX = 0; ffY = 0; ffZ = 0;
	targetX = 0;
	foods.length = 0; particles.length = 0;
}

/** Current widget height in character rows (scene + 4 status bar lines).
 *  Returns a live value since H can change when renderLumo resizes. */
export function lumoHeight(): number { return H + 4; }
/** @deprecated Use lumoHeight() — this constant is stale after resize. */
export const LUMO_HEIGHT = H + 4;
