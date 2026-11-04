// Showdown의 실제 데미지 계산(최대 난수, 급소 없음)을 시나리오 JSON 줄(stdin) -> 결과 JSON 줄(stdout)로 돌려줌 (tools/matchup_check.py가 사용).
// 시나리오: {att, def: {species, ability, item, status, boosts[5: atk def spa spd spe], stats[6: hp atk def spa spd spe], hp(비율), tera(타입 이름 또는 null)}, move, weather, terrain, screens:[reflect, lightscreen, auroraveil] (방어측), variant: 0 전체 | 1 필드 효과 없음 | 2 필드+도구 없음}
// 결과: {dmg(최대 데미지, 0이면 면역/무효/고정 피해), maxhp, dyn(위력 변동/고정 피해/특수 방어 규칙 기술이면 이유 문자열), hits}
const path = require('path');
const P = path.join(__dirname, '..', 'pokemon-showdown', 'dist');
const {Battle} = require(path.join(P, 'sim/battle'));
const {Dex} = require(path.join(P, 'sim'));
const STAT = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const mk = (s, item) => ({name: s.species, species: s.species, ability: s.ability || 'noability', item: item ? s.item : '', moves: ['tackle'], nature: 'Serious', level: 100,
  evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0}, ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}, teraType: s.tera || undefined});
function fit(p, s) {
  STAT.forEach((k, i) => { if (k === 'hp') { p.maxhp = p.baseMaxhp = s.stats[0]; p.hp = Math.max(1, Math.round(s.hp * s.stats[0])); } else p.storedStats[k] = p.baseStoredStats[k] = s.stats[i]; });
  ['atk', 'def', 'spa', 'spd', 'spe'].forEach((k, i) => { p.boosts[k] = Math.round(s.boosts[i]); });
  if (s.status) p.status = s.status;
  if (s.tera) { p.terastallized = s.tera; }
}
require('readline').createInterface({input: process.stdin}).on('line', line => {
  const sc = JSON.parse(line);
  try {
    const item = sc.variant < 2;
    const b = new Battle({formatid: 'gen9customgame@@@!teampreview', seed: [1, 2, 3, 4], p1: {name: 'a', team: [mk(sc.att, item)]}, p2: {name: 'b', team: [mk(sc.def, item)]}});
    b.randomizer = x => x;
    const a = b.p1.pokemon[0], d = b.p2.pokemon[0];
    fit(a, sc.att); fit(d, sc.def);
    b.field.clearWeather(); b.field.clearTerrain();                                  // 입장 특성(그래스메이커/가뭄/부스터에너지 등)이 건 효과는 시나리오 입력이 아니라 제거
    [a, d].forEach(p => { p.removeVolatile('protosynthesis'); p.removeVolatile('quarkdrive'); });
    if (sc.variant < 1) {
      if (sc.weather) b.field.setWeather(sc.weather, a);
      if (sc.terrain) b.field.setTerrain(sc.terrain, a);
      ['reflect', 'lightscreen', 'auroraveil'].forEach((c, i) => { if (sc.screens[i]) d.side.addSideCondition(c, a); });
    }
    const m = Dex.moves.get(sc.move);
    const mv = b.dex.getActiveMove(sc.move); mv.willCrit = false;
    const dyn = [m.basePowerCallback && 'bp-callback', m.damage && 'fixed', m.overrideOffensiveStat && 'off-stat', m.overrideDefensiveStat && 'def-stat', m.overrideOffensivePokemon && 'off-mon',
      m.ohko && 'ohko', m.multihit && 'multihit', m.onModifyType && 'type-change', m.onModifyMove && 'mod-move', m.onBasePower && 'base-power-hook', m.onModifyAtk && 'x'].filter(Boolean).join(',');
    const dmg = b.actions.getDamage(a, d, mv, true);
    out(sc.id, {dmg: dmg || 0, maxhp: d.maxhp, dyn, hits: Array.isArray(m.multihit) ? m.multihit : (m.multihit || 1)});
  } catch (e) { out(sc.id, {err: String(e).slice(0, 120)}); }
});
function out(id, o) { console.log(JSON.stringify(Object.assign({id}, o))); }
