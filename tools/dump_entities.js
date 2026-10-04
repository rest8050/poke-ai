// Showdown 데이터에서 도구/특성/기술의 이름·설명·훅·플래그를 JSON으로 내보냄 (build_entity_features.py가 사용)
const path = require('path'), fs = require('fs');
const P = path.join(__dirname, '..', 'pokemon-showdown', 'dist');
const {Dex} = require(path.join(P, 'sim'));
const T = {items: require(path.join(P, 'data/text/items.js')).ItemsText,
           abilities: require(path.join(P, 'data/text/abilities.js')).AbilitiesText,
           moves: require(path.join(P, 'data/text/moves.js')).MovesText};
const vocab = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'vocab.json'), 'utf8'));
const BOOL = ['isBerry', 'isChoice', 'isGem', 'isPokeball', 'isPrimalOrb'];
function dump(dex, text, ids) {
  const out = [];
  for (const id of ids) {
    const e = dex.get(id); if (!e.exists) continue;
    const t = text[id] || {};
    const hooks = Object.keys(e).filter(k => /^on[A-Z]/.test(k) && e[k] != null && !/(Priority|Order|SubOrder)$/.test(k));
    const flags = {}; for (const f of BOOL) if (e[f]) flags[f] = 1;
    Object.assign(flags, e.flags || {});
    out.push({id, name: e.name, desc: t.shortDesc || t.desc || '', hooks, flags});
  }
  return out;
}
fs.writeFileSync(process.argv[2], JSON.stringify({
  item: dump(Dex.items, T.items, Object.keys(vocab.item)),
  ability: dump(Dex.abilities, T.abilities, Object.keys(vocab.ability)),
  move: dump(Dex.moves, T.moves, Object.keys(vocab.move)),
}));
