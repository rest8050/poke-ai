// 장면 생성기: 양쪽 행동을 스크립트로 고정해 Showdown 시뮬레이터로 위치까지 진행하고, p1 시점 메시지를 foul-play/fp/replay_label.py가 읽는 rec 형식(jsonl)으로 저장.
// 마지막 단계가 끝난 뒤 p1의 결정 요청이 대기 중인 상태가 테스트 장면 (replay_label은 그 결정에서 FP 탐색 분포를 남김).
// 입력(JSON 파일): [{id, p1:{name, team(export 텍스트)}, p2:{name, team}, script:[{p1:"team 1"|"move 2"|"move 3 terastallize"|"switch 4", p2:...}, ...], seed:[a,b,c,d]}]
//   단계에 reseed:[a,b,c,d]를 넣으면 그 단계 직전에 난수 상태를 교체. 단계에서 한쪽만 적으면 그쪽만 선택 (기절 뒤 강제 교체). 선택이 거절되면 오류 줄을 남기고 그 장면은 버림.
// 출력(jsonl, 한 줄 = 한 장면): rec 필드(tag,user,export,frames) + id + state(p1/p2 포켓몬 상태, 턴) + log(omniscient 로그, 검증용)
// 사용: node tools/scenario_gen.js <입력.json> <출력.jsonl>
const fs = require('fs');
const path = require('path');
const P = path.join(__dirname, '..', 'pokemon-showdown', 'dist');
const {BattleStream, Teams, PRNG} = require(path.join(P, 'sim'));

async function run(sc, n) {
  const tag = `battle-gen9ou-${900000 + n}`;
  const stream = new BattleStream();
  const out = {log: [], end: false};
  let rq = 0;
  const frames = [`>${tag}\n|init|battle\n|title|${sc.p1.name} vs. ${sc.p2.name}\n|j|☆${sc.p1.name}\n|j|☆${sc.p2.name}`];
  const buf = [];
  (async () => { for await (const c of stream) buf.push(c); })();           // 읽기는 한 곳에서만 (경쟁 읽기는 청크를 잃음)
  const drain = async () => {                     // 입력이 처리돼 청크가 더 안 나올 때까지 기다려 모음
    let n = -1;
    while (n !== buf.length) { n = buf.length; await new Promise(res => setTimeout(res, 40)); }
    return buf.splice(0);
  };
  const {extractChannelMessages} = require(path.join(P, 'sim', 'battle'));
  const handle = got => {
    const ups = [], reqs = [];
    for (const chunk of got) {
      const i = chunk.indexOf('\n'), type = chunk.slice(0, i), data = chunk.slice(i + 1);
      if (type === 'update') {
        const m = extractChannelMessages(data, [-1, 1]);
        ups.push(m[1].join('\n')); out.log.push(...m[-1]);
      } else if (type === 'sideupdate') {
        const j = data.indexOf('\n'), side = data.slice(0, j);
        if (side === 'p1') reqs.push(data.slice(j + 1));
      } else if (type === 'end') out.end = true;
    }
    for (const u of ups) if (u.trim()) frames.push(`>${tag}\n${u}`);       // 갱신을 먼저, 요청은 뒤에 (실제 웹소켓 순서)
    for (const r of reqs) {                                                   // 실제 서버 요청에는 rqid가 있고 Foul Play가 읽음
      const q = JSON.parse(r.replace(/^\|request\|/, ''));
      q.rqid = ++rq;
      frames.push(`>${tag}\n|request|${JSON.stringify(q)}`);
    }
  };
  const seed = sc.seed || [1, 2, 3, 4];
  const team = t => Teams.pack(Teams.import(t));
  stream.write(`>start ${JSON.stringify({formatid: 'gen9ou', seed})}`);
  stream.write(`>player p1 ${JSON.stringify({name: sc.p1.name, team: team(sc.p1.team)})}`);
  stream.write(`>player p2 ${JSON.stringify({name: sc.p2.name, team: team(sc.p2.team)})}`);
  handle(await drain());
  for (const [k, step] of sc.script.entries()) {
    if (step.reseed) stream.battle.prng = new PRNG(step.reseed);              // 이 단계의 턴 난수만 바꿔 같은 위치에서 결과를 다시 뽑음 (반사실 검증용)
    for (const s of ['p1', 'p2']) if (step[s]) stream.write(`>${s} ${step[s]}`);
    handle(await drain());
    if (out.end) throw new Error(`배틀이 단계 ${k}에서 끝남`);
  }
  const b = stream.battle;
  const bad = b.inputLog.filter(l => /error|invalid/i.test(l));
  const mon = p => ({name: p.name, species: p.species.name, hp: p.hp, maxhp: p.maxhp, status: p.status || '', fainted: p.fainted, active: p.isActive, item: p.item, ability: p.ability, tera: p.terastallized || '', boosts: {...p.boosts}});
  const state = {turn: b.turn, p1: b.sides[0].pokemon.map(mon), p2: b.sides[1].pokemon.map(mon), p1Req: b.sides[0].activeRequest ? Object.keys(b.sides[0].activeRequest) : null, weather: b.field.weather, terrain: b.field.terrain};
  if (!state.p1Req) throw new Error('p1 결정 요청이 대기 중이 아님 (스크립트 선택이 거절됐을 수 있음)');
  frames.push(`>${tag}\n|\n|win|${sc.p2.name}`, `>${tag}\n|deinit`);         // 재생이 정상 종료되게 마무리 (마지막 결정 이후는 라벨에 무관)
  return {tag, user: sc.p1.name, export: sc.p1.team, frames, id: sc.id, state, log: out.log};
}

(async () => {
  const sc = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const lines = [];
  for (const [n, s] of sc.entries()) {
    try { lines.push(JSON.stringify(await run(s, n))); }
    catch (e) { console.error(`장면 ${s.id} 실패: ${e.message}`); }
  }
  fs.writeFileSync(process.argv[3], lines.join('\n') + (lines.length ? '\n' : ''));
  console.log(`${lines.length}/${sc.length} 장면 저장 -> ${process.argv[3]}`);
  process.exit(0);
})();
