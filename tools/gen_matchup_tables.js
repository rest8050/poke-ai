// Showdown 데이터에서 매치업 모듈(src/core/matchup_v7.py)이 쓰는 기술 태그/진화 전 종/종 기본 타입 표를 만든다 (관측의 타입 칸은 테라하면 테라 타입으로 덮여서 원래 타입이 사라짐) -> data/matchup_tables.json
// 사용: node tools/gen_matchup_tables.js (vocab.json의 기술/종만)
const fs = require('fs'), path = require('path');
const {Dex} = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));
const V = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'vocab.json'), 'utf8'));
const FLAG = ['slicing', 'punch', 'sound', 'contact', 'bite', 'pulse'];
const moves = {};
for (const id of Object.keys(V.move)) {
  const m = Dex.moves.get(id);
  if (!m.exists || m.category === 'Status') continue;
  const tags = FLAG.filter(f => m.flags[f]);
  if (m.recoil || m.hasCrashDamage) tags.push('recoil');
  if (m.secondaries && m.secondaries.length || m.secondary) tags.push('secondary');
  if (m.overrideOffensiveStat === 'def') tags.push('bodypress');
  if (m.overrideOffensivePokemon === 'target') tags.push('foulplay');
  if (m.overrideDefensiveStat === 'def') tags.push('psyshock');
  if (tags.length) moves[id] = tags;
}
const nfe = Object.keys(V.species).filter(id => { const s = Dex.species.get(id); return s.exists && s.nfe; });
const types = {};
for (const id of Object.keys(V.species)) { const s = Dex.species.get(id); if (s.exists) types[id] = s.types.map(t => t.toLowerCase()); }
fs.writeFileSync(path.join(__dirname, '..', 'data', 'matchup_tables.json'), JSON.stringify({moves, nfe, types}));
console.log(Object.keys(moves).length, 'moves tagged,', nfe.length, 'nfe species');
