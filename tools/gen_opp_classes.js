// 상대 응수 클래스 테이블: 어휘의 기술 이름 -> 클래스 번호 (data/opp_classes.json). 0 공격, 1 우선도 공격, 2 피벗 공격, 3 셋업, 4 회복, 5 방어, 6 설치(해저드/제거/필드), 7 상태이상 부여, 8 기타 변화(피벗 변화 포함); 9는 교체(기술이 아님).
// 기반은 data/move_classes.json(11종), 공격 중 Showdown 우선도 > 0이면 1로 따로 뺌 (기습/전광석화/막치기 등: 교체로 무력화되는 응수가 핵심이라 구분).
// 사용: node tools/gen_opp_classes.js
const fs = require('fs');
const path = require('path');
const {Dex} = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));
const vocab = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'vocab.json'), 'utf8'));
const cls = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'move_classes.json'), 'utf8'));
const BASE = {attack: 0, pivot_attack: 2, setup: 3, recovery: 4, protect: 5, hazard: 6, hazard_removal: 6, field_setup: 6, status_inflict: 7, status_other: 8, pivot_status: 8};
const out = {};
const count = Array(10).fill(0);
for (const name of Object.keys(vocab.move)) {
  const c = BASE[cls[name]];
  if (c === undefined) continue;
  const m = Dex.moves.get(name);
  const k = c === 0 && m.exists && m.priority > 0 && m.category !== 'Status' ? 1 : c;
  out[name] = k; count[k]++;
}
fs.writeFileSync(path.join(__dirname, '..', 'data', 'opp_classes.json'), JSON.stringify(out));
console.log('기술', Object.keys(out).length, '클래스별', count.join(' '));
