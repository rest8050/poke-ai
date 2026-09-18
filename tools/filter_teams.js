/**
 * 팀 파일 폴더를 로컬 Showdown 검증기로 일괄 검사해 합법 팀만 팀 풀 JSON으로 저장.
 * 학습용 / 평가 전용(holdout) 두 파일로 나눔. 형식은 data/team_pool.json 과 같음 ({"teams": [{name, export}]}).
 *
 * 사용 (poke-ai 폴더에서):
 *   node tools/filter_teams.js --dir data/metamon/teams/gen9ou
 *   node tools/filter_teams.js --dir data/metamon/teams/gen9ou --format gen9ou --holdout 500 --seed 42 --limit 1000
 *
 * 출력: data/team_pool_metamon.json, data/team_pool_metamon_holdout.json, 콘솔에 탈락 사유 집계
 */
const fs = require('fs');
const path = require('path');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
	if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
	return acc;
}, []));
const DIR = args.dir || 'data/metamon/teams/gen9ou';
const FORMAT = args.format || 'gen9ou';
const OUT = args.out || 'data/team_pool_metamon.json';
const HOLDOUT_OUT = args['holdout-out'] || OUT.replace(/\.json$/, '_holdout.json');
const HOLDOUT = Number(args.holdout ?? 500);
const SEED = Number(args.seed ?? 42);
const LIMIT = args.limit ? Number(args.limit) : Infinity;

const {Teams, TeamValidator} = require(path.resolve('pokemon-showdown/dist/sim'));
const validator = TeamValidator.get(FORMAT);

// 검증기 동작 확인: 합법 팀은 통과, 수면기 조항 위반(버섯포자)은 탈락해야 함
const LEGAL = `Amoonguss @ Black Sludge
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 148 Def / 108 SpD
Sassy Nature
- Clear Smog
- Giga Drain
- Foul Play
- Sludge Bomb`;
if (validator.validateTeam(Teams.import(LEGAL)) !== null) throw new Error('self-check: 합법 팀이 탈락함');
if (validator.validateTeam(Teams.import(LEGAL.replace('- Giga Drain', '- Spore'))) === null) {
	throw new Error('self-check: 불법 팀(Spore)이 통과함');
}

function mulberry32(a) {
	return () => {
		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul(a ^ a >>> 15, 1 | a);
		t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
		return ((t ^ t >>> 14) >>> 0) / 4294967296;
	};
}

const files = fs.readdirSync(DIR).filter(f => f.endsWith(`.${FORMAT}_team`)).sort().slice(0, LIMIT);
const seen = new Set();
const valid = [];
const reasons = new Map();
let duplicates = 0, unparsable = 0;
const t0 = Date.now();

for (const [i, file] of files.entries()) {
	const text = fs.readFileSync(path.join(DIR, file), 'utf8').trim();
	const sets = Teams.import(text);
	if (!sets || sets.length !== 6) {
		unparsable++;
		continue;
	}
	const packed = Teams.pack(sets);
	if (seen.has(packed)) {
		duplicates++;
		continue;
	}
	seen.add(packed);
	const problems = validator.validateTeam(sets);
	if (problems) {
		// 사유 집계: 포켓몬/기술 이름이 섞인 문장이라 첫 문제만 그대로 셈
		const key = problems[0].replace(/\s+/g, ' ').slice(0, 90);
		reasons.set(key, (reasons.get(key) || 0) + 1);
		continue;
	}
	valid.push({name: path.basename(file).replace(/\.[^.]+$/, ''), export: text});
	if ((i + 1) % 5000 === 0) console.log(`  ${i + 1}/${files.length} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}

// 결정적 셔플 후 앞에서 holdout 분리
const rand = mulberry32(SEED);
for (let i = valid.length - 1; i > 0; i--) {
	const j = Math.floor(rand() * (i + 1));
	[valid[i], valid[j]] = [valid[j], valid[i]];
}
const holdout = valid.slice(0, Math.min(HOLDOUT, Math.floor(valid.length / 2)));
const train = valid.slice(holdout.length);

const meta = {source: DIR, format: FORMAT, seed: SEED, created: new Date().toISOString()};
fs.writeFileSync(OUT, JSON.stringify({...meta, split: 'train', teams: train}, null, 1));
fs.writeFileSync(HOLDOUT_OUT, JSON.stringify({...meta, split: 'holdout', teams: holdout}, null, 1));

const failed = [...reasons.values()].reduce((a, b) => a + b, 0);
console.log(`\n검사 ${files.length}개 (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
console.log(`  합법 ${valid.length} | 불법 ${failed} | 중복 ${duplicates} | 파싱 실패 ${unparsable}`);
console.log(`  학습용 ${train.length} → ${OUT}`);
console.log(`  평가 전용 ${holdout.length} → ${HOLDOUT_OUT}`);
console.log('\n탈락 사유 상위 15:');
for (const [k, v] of [...reasons.entries()].sort((a, b) => b[1] - a[1]).slice(0, 15)) console.log(`  ${String(v).padStart(6)}  ${k}`);
