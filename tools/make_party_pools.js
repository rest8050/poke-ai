/**
 * "처음 다뤄보는 파티" 평가용 팀 풀 두 개를 만든다 (형식은 data/team_pool_metamon_holdout.json과 같음).
 *   B 희귀 파티   : 학습 팀 풀에서 거의 안 쓰인 포켓몬 6마리 (세트는 실제 사람 팀에 있던 그대로)
 *   C 랜덤 세트 파티: 랜덤배틀 세트 데이터로 만든 무작위 세트 6마리 (도구는 사용률 가중 무작위, 노력치는 단순 규칙)
 * 모든 팀은 로컬 Showdown 검증기(gen9ou)를 통과한 팀만 저장. 각 풀이 기존 holdout(A)과 얼마나 다른지 통계도 출력.
 *
 * 사용 (poke-ai 폴더에서): node tools/make_party_pools.js [--n 500] [--seed 7] [--rare-min 3] [--rare-max 100]
 * 출력: data/team_pool_rare.json, data/team_pool_randomset.json  (평가 전용: 학습에 쓰지 말 것. 같은 시드면 항상 같은 팀이 나옴, 다른 시드로 덮어쓰기는 --force 없이는 거부)
 * 학습용(평가 풀과 팀이 겹치지 않음):
 *   node tools/make_party_pools.js --seed 11 --suffix _train --plain --avoid data/team_pool_rare.json,data/team_pool_randomset.json --train-a 2000
 *   → data/team_pool_rare_train.json, data/team_pool_randomset_train.json, data/team_pool_train_a.json (학습 팀 풀에서 무작위 N팀)
 *   새 풀 추가: --train-a-out data/team_pool_train_a2.json 로 A 출력 파일명을 바꾸고, --avoid에 이전 학습 풀들을 넣으면 겹치지 않는 팀만 뽑음
 */
const fs = require('fs');
const path = require('path');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
	if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]]);
	return acc;
}, []));
const N = Number(args.n ?? 500), SEED = Number(args.seed ?? 7);
const RARE_MIN = Number(args['rare-min'] ?? 3), RARE_MAX = Number(args['rare-max'] ?? 100);
const FORMAT = args.format ?? 'gen9ou';   // 예: gen9anythinggoes (Uber/AG 종 커버용)
const UBER_MIN = Number(args['uber-min'] ?? 0);   // randomset 팀에 OU 금지(Uber/AG 티어) 종을 최소 몇 마리 넣을지
const ONLY_RANDOMSET = args['only-randomset'] === '1';
const SUFFIX = args.suffix ?? '';
const AVOID = (args.avoid ?? '').split(',').filter(Boolean);

const {Teams, TeamValidator, Dex} = require(path.resolve('pokemon-showdown/dist/sim'));
const validator = TeamValidator.get(FORMAT);

function mulberry32(a) {
	return () => {
		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul(a ^ a >>> 15, 1 | a);
		t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
		return ((t ^ t >>> 14) >>> 0) / 4294967296;
	};
}
const rnd = mulberry32(SEED);
const pick = arr => arr[Math.floor(rnd() * arr.length)];
function sample(arr, k) {
	const a = arr.slice();
	for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
	return a.slice(0, k);
}
const readPool = f => JSON.parse(fs.readFileSync(f, 'utf8').replace(/^﻿/, ''));
const setKey = s => Teams.export([s]).trim();

// ---- 학습 팀 풀: 종별 사용 횟수와 실제 세트 ----
const trainSets = new Map();      // 종 → [세트]
const trainCount = new Map();     // 종 → 횟수
const trainSetKeys = new Set();
const itemCount = new Map();
for (const t of readPool('data/team_pool_metamon.json').teams) {
	for (const s of Teams.import(t.export) || []) {
		trainCount.set(s.species, (trainCount.get(s.species) || 0) + 1);
		const k = setKey(s);
		if (!trainSetKeys.has(k)) {
			trainSetKeys.add(k);
			if (!trainSets.has(s.species)) trainSets.set(s.species, []);
			trainSets.get(s.species).push(s);
		}
		if (s.item) itemCount.set(s.item, (itemCount.get(s.item) || 0) + 1);
	}
}
const rank = new Map([...trainCount.entries()].sort((a, b) => b[1] - a[1]).map(([sp], i) => [sp, i / trainCount.size]));

// ---- B: 희귀 종 6마리 ----
const rareSpecies = [...trainCount].filter(([, c]) => c >= RARE_MIN && c <= RARE_MAX).map(([sp]) => sp);
console.log(`B: 희귀 종 ${rareSpecies.length}개 (학습 팀 풀 사용 ${RARE_MIN}~${RARE_MAX}회 / 전체 ${trainCount.size}종)`);

// ---- C: 랜덤배틀 세트 ----
const rb = JSON.parse(fs.readFileSync('pokemon-showdown/data/random-battles/gen9/sets.json', 'utf8'));
const itemList = [...itemCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 30);
const itemTotal = itemList.reduce((a, [, c]) => a + c, 0);
function randomItem() {
	let r = rnd() * itemTotal;
	for (const [it, c] of itemList) { if ((r -= c) <= 0) return it; }
	return itemList[0][0];
}
function randomSet(name, entry) {
	const role = pick(entry.sets);
	const sp = Dex.species.get(name);
	const phys = sp.baseStats.atk >= sp.baseStats.spa;
	// 역할의 기술 목록이 4개 미만이면 같은 종의 다른 역할 목록까지 합쳐서 채움 (기술 4개 미만인 포켓몬은 poke-engine이 패닉함)
	let pool = role.movepool;
	if (pool.length < 4) pool = [...new Set(entry.sets.flatMap(s => s.movepool))];
	return {
		name: '', species: sp.name, item: randomItem(), ability: pick(role.abilities), gender: '',
		moves: sample(pool, Math.min(4, pool.length)), nature: phys ? 'Jolly' : 'Timid',
		evs: phys ? {hp: 4, atk: 252, def: 0, spa: 0, spd: 0, spe: 252} : {hp: 4, atk: 0, def: 0, spa: 252, spd: 0, spe: 252},
		ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}, level: 100, teraType: pick(role.teraTypes),
	};
}
// 종 자체가 OU에서 합법인 것만 (세트 몇 번 시도해 한 번이라도 통과하면 합법으로 봄)
const legalRb = [];
for (const id of Object.keys(rb)) {
	for (let i = 0; i < 4; i++) {
		if (validator.validateTeam([randomSet(id, rb[id])]) === null) { legalRb.push(id); break; }
	}
}
console.log(`C: 랜덤배틀 세트가 있는 ${Object.keys(rb).length}종 중 OU 합법 ${legalRb.length}종`);

const norm = t => t.replace(/\s+/g, ' ').trim();
const avoided = new Set();
for (const f of AVOID) for (const t of readPool(f).teams) avoided.add(norm(t.export));
function build(kind, makeTeam) {
	const teams = [], seen = new Set(avoided);
	let tries = 0;
	while (teams.length < N && tries++ < N * 60) {
		const team = makeTeam();
		if (!team || team.some(s => s.moves.length !== 4) || validator.validateTeam(team) !== null) continue;
		if (args.plain) for (const s of team) { s.gender = ''; s.name = ''; }  // 성별/별명 머리글 제거 (Foul Play·poke-env 파서가 '(M)'을 종 이름으로 오해하는 문제 방지)
		const text = Teams.export(team);
		if (seen.has(text)) continue;
		seen.add(text);
		teams.push({name: `${kind}-${String(teams.length).padStart(3, '0')}`, export: text});
	}
	console.log(`${kind}: 합법 팀 ${teams.length}/${N}개 (시도 ${tries})`);
	return teams;
}

const rare = ONLY_RANDOMSET ? [] : build('rare', () => {
	const sps = sample(rareSpecies, 6);
	return sps.map(sp => pick(trainSets.get(sp)));
});
const isUberTier = id => ['Uber', 'AG'].includes(Dex.species.get(id).tier);
const uberIds = legalRb.filter(isUberTier), restIds = legalRb.filter(id => !isUberTier(id));
if (UBER_MIN) console.log(`C: ${FORMAT} 합법 종 중 OU 금지(Uber/AG) ${uberIds.length}종, 그 외 ${restIds.length}종`);
const randomset = build('randomset', () => {
	const ids = UBER_MIN ? [...sample(uberIds, UBER_MIN), ...sample(restIds, 6 - UBER_MIN)] : sample(legalRb, 6);
	return ids.map(id => randomSet(id, rb[id]));
});

// ---- 통계: 기존 holdout(A)과 비교 ----
function stats(name, teams) {
	let n = 0, seenSet = 0, zero = 0, pctSum = 0, pctN = 0;
	for (const t of teams) {
		for (const s of Teams.import(t.export)) {
			n++;
			if (trainSetKeys.has(setKey(s))) seenSet++;
			if (!trainCount.has(s.species)) zero++;
			else { pctSum += rank.get(s.species); pctN++; }
		}
	}
	console.log(`  ${name.padEnd(10)} 팀 ${String(teams.length).padStart(4)} | 학습에 있던 동일 세트 ${(100 * seenSet / n).toFixed(1)}% | 학습 팀 풀에 없는 종 ${(100 * zero / n).toFixed(1)}% | 종 사용률 순위 평균 ${(100 * pctSum / pctN).toFixed(0)}%(0=최다 사용, 100=최소)`);
}
console.log('\n기존 평가 풀(A)과 비교:');
stats('A holdout', readPool('data/team_pool_metamon_holdout.json').teams);
if (!ONLY_RANDOMSET) stats('B rare', rare);
stats('C random', randomset);

const meta = (split, teams) => ({
	source: 'tools/make_party_pools.js', format: FORMAT, seed: SEED, created: new Date().toISOString(), split, teams,
});
function write(file, split, teams) {
	if (fs.existsSync(file) && !args.force && readPool(file).seed !== SEED) {
		throw new Error(`${file}은 다른 시드(${readPool(file).seed})로 만든 풀이라 덮어쓰지 않음 (--force로 강제, 평가 풀이 바뀌니 주의)`);
	}
	fs.writeFileSync(file, JSON.stringify(meta(split, teams), null, 1));
	console.log(`저장: ${file} (${teams.length}팀)`);
}
if (!ONLY_RANDOMSET) write(`data/team_pool_rare${SUFFIX}.json`, 'rare', rare);
write(`data/team_pool_randomset${SUFFIX}.json`, 'randomset', randomset);
if (args['train-a']) {  // 학습용 A: 학습 팀 풀(평가 holdout과 별개)에서 무작위 N팀
	const all = readPool('data/team_pool_metamon.json').teams.filter(t => !avoided.has(norm(t.export)));  // --avoid에 준 풀(이전 train_a 포함)과 겹치지 않게
	const aOut = args['train-a-out'] ?? 'data/team_pool_train_a.json';
	write(aOut, 'train_a', sample(all, Number(args['train-a'])).map((t, k) => ({name: `train_a-${String(k).padStart(4, '0')}`, export: t.export})));
}
