// 반사실 정답 생성기 (M5, docs/chain_design.md): 임의 팀 + 임의 합법 수 접두로 위치를 만들고, 그 위치에서
// (내 모든 합법 행동 a) x (상대 응수 클래스별 대표 수 o)를 시뮬레이터로 한 턴씩 R번(턴 안 난수만 바꿔서) 돌려 한 턴 뒤 상태를 남긴다.
// 입력(JSON): {n: 위치 수, seed, reps, pool: 팀 풀 json 경로, maxPrefix: 접두 최대 턴, minPrefix(선택), noSteps(선택: 동치 검증용 단계 기록 생략), fromFile(선택: 이 위치 파일을 복원해 칸만 다시 계산), classMean(선택: 응수 클래스 안 모든 기술/교체 대상 평균)}  출력(jsonl, 한 줄 = 한 위치)
//   {id, p1, p2 (export), seed, prefix:[{p1,p2}...] (장면 정의의 script), pre: 위치 상태, acts:[{choice,kind,move,tera,sp}], opps:[{cls,choice,kind,move,sp}],
//    rseeds:[[4개]...], cells:[{a,o,reps:[post 상태], steps0:[{p1,p2}...] (rep 0의 위치 뒤 단계, 동치 검증용)}]}
// 상태 = {turn, req: "move"|"switch"|"end", p1:[{sp,hp,mhp,f,act,bo[5]}x6], p2:[...]} (포켓몬 순서는 시뮬레이터 팀 순서)
// 사용: node tools/cf_gen.js <cfg.json> <out.jsonl>
const fs = require('fs');
const path = require('path');
const P = path.join(__dirname, '..', 'pokemon-showdown', 'dist');
const {Battle, Teams, PRNG} = require(path.join(P, 'sim'));
const CLS = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'opp_classes.json'), 'utf8'));
const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const pool = !cfg.pool ? [] : JSON.parse(fs.readFileSync(path.resolve(__dirname, '..', cfg.pool), 'utf8').replace(/^﻿/, '')).teams.map(t => t.export);

function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rnd = mulberry(cfg.seed);
const pick = a => a[Math.floor(rnd() * a.length)];
const pack = t => Teams.pack(Teams.import(t));
const sp = p => p.details.split(',')[0];

function legal(b, si) {                       // 대기 중인 요청에서 합법 행동 목록. 요청 없으면 null
  const r = b.sides[si].activeRequest;
  if (!r || r.wait) return null;
  if (r.teamPreview) return {team: true, acts: []};
  const pk = r.side.pokemon, sw = [];
  pk.forEach((p, i) => { if (!p.active && !p.condition.endsWith(' fnt')) sw.push({choice: `switch ${i + 1}`, kind: 'switch', sp: sp(p)}); });
  if (r.forceSwitch) return {force: true, acts: sw};
  const a = r.active[0], acts = [];
  a.moves.forEach((m, i) => { if (!m.disabled) acts.push({choice: `move ${i + 1}`, kind: 'move', move: m.id, tera: false}); });
  if (a.canTerastallize) a.moves.forEach((m, i) => { if (!m.disabled) acts.push({choice: `move ${i + 1} terastallize`, kind: 'move', move: m.id, tera: true}); });
  if (!a.trapped) acts.push(...sw);
  return {acts};
}

function randomChoice(L) {                     // 접두용 임의 정책: 기술 위주, 가끔 교체/테라
  if (L.team) return `team ${1 + Math.floor(rnd() * 6)}`;
  const mv = L.acts.filter(a => a.kind === 'move' && !a.tera), te = L.acts.filter(a => a.tera), sw = L.acts.filter(a => a.kind === 'switch');
  const u = rnd();
  const pool_ = u < 0.15 && sw.length ? sw : u < 0.22 && te.length ? te : mv.length ? mv : sw;
  return pick(pool_).choice;
}

function pending(b) { return [0, 1].map(si => legal(b, si)); }

function round(b, steps) {                     // 요청이 있는 쪽 모두 임의로 고르고 확정
  const L = pending(b), step = {};
  L.forEach((l, si) => { if (l) step[si ? 'p2' : 'p1'] = randomChoice(l); });
  for (const s of ['p1', 'p2']) if (step[s]) b.choose(s, step[s]);
  steps.push(step);
}

const mon = p => ({sp: p.set.species, hp: p.hp, mhp: p.maxhp, f: p.fainted ? 1 : 0, act: p.isActive ? 1 : 0, bo: ['atk', 'def', 'spa', 'spd', 'spe'].map(k => p.boosts[k])});
function state(b) {
  const r = b.sides[0].activeRequest;
  const req = b.ended ? 'end' : r && !r.wait ? (r.forceSwitch ? 'switch' : 'move') : 'wait';
  return {turn: b.turn, req, p1: b.sides[0].pokemon.map(mon), p2: b.sides[1].pokemon.map(mon)};
}
const p1Normal = b => { const l = legal(b, 0); return !b.ended && l && !l.team && !l.force; };

function position(id, src) {                   // src: 이미 만든 위치(cfg.fromFile)면 접두 단계를 그대로 재생해 같은 위치를 복원
  let t1, t2, seed, steps = [], b;
  if (src) {
    ({p1: t1, p2: t2, seed} = src);
    b = new Battle({formatid: 'gen9ou', seed, p1: {name: 'scn1', team: pack(t1)}, p2: {name: 'scn2', team: pack(t2)}});
    for (const st of src.prefix) { for (const s_ of ['p1', 'p2']) if (st[s_]) b.choose(s_, st[s_]); steps.push(st); }
  } else {
    [t1, t2] = [pick(pool), pick(pool)];
    seed = [1 + Math.floor(rnd() * 1e6), 2 + Math.floor(rnd() * 1e6), 3 + Math.floor(rnd() * 1e6), 4 + Math.floor(rnd() * 1e6)];
    b = new Battle({formatid: 'gen9ou', seed, p1: {name: 'scn1', team: pack(t1)}, p2: {name: 'scn2', team: pack(t2)}});
    round(b, steps);                                                     // 팀 프리뷰
    const lo = cfg.minPrefix || 0, turns = lo + Math.floor(rnd() * (cfg.maxPrefix - lo + 1));
    for (let k = 0; k < turns && !b.ended; k++) round(b, steps);
    for (let k = 0; k < 6 && !b.ended && !p1Normal(b); k++) round(b, steps);   // 강제 교체 등을 처리해 p1의 일반 결정에서 멈춤
  }
  if (b.ended || !p1Normal(b)) return null;
  const A = legal(b, 0).acts;
  const L2 = legal(b, 1);
  if (!L2 || L2.team || L2.force) return null;
  const opps = [], seen = new Set();
  for (const a of L2.acts.filter(x => x.kind === 'move' && !x.tera)) {  // 클래스별 첫 기술 (classMean이면 그 클래스의 모든 기술을 members에 담아 평균)
    const c = CLS[a.move];
    if (c !== undefined && !seen.has(c)) { seen.add(c); opps.push({cls: c, ...a, members: cfg.classMean ? L2.acts.filter(x => x.kind === 'move' && !x.tera && CLS[x.move] === c) : undefined}); }
  }
  const sw2 = L2.acts.filter(a => a.kind === 'switch');
  if (sw2.length) opps.push(cfg.classMean ? {cls: 9, ...sw2[0], members: sw2} : {cls: 9, ...pick(sw2)});
  if (!A.length || !opps.length) return null;
  const rseeds = src ? src.rseeds.slice(0, cfg.reps) : Array.from({length: cfg.reps}, () => [1 + Math.floor(rnd() * 1e6), 2 + Math.floor(rnd() * 1e6), 3 + Math.floor(rnd() * 1e6), 4 + Math.floor(rnd() * 1e6)]);
  const snap = Battle.fromJSON(b.toJSON());
  const cells = [];
  A.forEach((a, ai) => opps.forEach((o, oi) => {
    const reps = []; let steps0 = null;
    (o.members || [o]).forEach(mem => rseeds.forEach((rs, ri) => {
      const c = Battle.fromJSON(JSON.stringify(snap.toJSON()));
      c.prng = new PRNG(rs);
      const st = [{p1: a.choice, p2: mem.choice}];
      c.choose('p1', a.choice); c.choose('p2', mem.choice);
      for (let k = 0; k < 4 && !c.ended; k++) {
        const r1 = c.sides[0].activeRequest;
        if (r1 && !r1.wait) break;
        const l2 = legal(c, 1);
        if (!l2) break;
        const ch = pick(l2.acts).choice;                                // p1은 대기 중 = 상대만 강제 교체
        c.choose('p2', ch); st.push({p2: ch});
      }
      reps.push(state(c));
      if (ri === 0 && !cfg.noSteps && !cfg.classMean) steps0 = st;
    }));
    cells.push({a: ai, o: oi, reps, steps0});
  }));
  return {id, p1: t1, p2: t2, seed, prefix: steps, pre: state(b), acts: A, opps, rseeds, cells};
}

const fd = fs.openSync(process.argv[3], 'w');
let n = 0;
const emit = p => { if (p) { fs.writeSync(fd, JSON.stringify(p) + '\n'); n++; } };
if (cfg.fromFile) {                            // 기존 위치 파일의 접두를 재생해 같은 위치에서 칸만 다시 계산 (classMean: 응수 클래스의 모든 기술 평균)
  for (const line of fs.readFileSync(path.resolve(__dirname, '..', cfg.fromFile), 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const src = JSON.parse(line);
    try { const p = position(src.id, src); if (p) { if (JSON.stringify(p.pre) !== JSON.stringify(src.pre)) throw new Error('복원한 위치가 다름'); emit(p); } } catch (e) { console.error(`위치 ${src.id} 실패: ${e.message}`); }
  }
} else {
  for (let i = 0; n < cfg.n && i < cfg.n * 4; i++) {
    try { emit(position(`cf${n}`)); } catch (e) { console.error(`위치 실패: ${e.message}`); }
  }
}
fs.closeSync(fd);
console.log(`${n} 위치 저장 -> ${process.argv[3]}`);
