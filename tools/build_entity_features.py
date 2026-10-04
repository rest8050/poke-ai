"""
도구/특성/기술의 이름·설명문·훅·플래그 → vocab ID 순서의 고정 특징 표 (data/entity_features.npz)

- 설명문 단어 집합(BoW, 최대 160) + 엔진 훅/플래그 멀티핫(최대 40)
- (bge-m3 텍스트 임베딩 PCA 96d는 제거: 이름/설명 의미 임베딩은 메커니즘과 무관하고 효과가 없었음)
- <kind>_seen: 학습 팀 풀(data/team_pool_metamon.json)에 등장한 횟수 (모델의 미학습 ID 행 0 처리에 사용)
사용: python tools/build_entity_features.py
"""
import collections, json, os, re, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, ".")
from sklearn.feature_extraction.text import CountVectorizer
from src.core.model import load_vocab_sizes

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
KINDS = {"item": "num_items", "ability": "num_abilities", "move": "num_moves"}


def pool_counts():
    cnt = {k: collections.Counter() for k in KINDS}
    for t in json.load(open("data/team_pool_metamon.json", encoding="utf-8-sig"))["teams"]:
        for blk in t["export"].strip().split("\n\n"):
            lines = blk.split("\n")
            m = re.match(r".* @ (.+?)\s*$", lines[0])
            if m: cnt["item"][norm(m.group(1))] += 1
            for l in lines:
                if l.startswith("Ability:"): cnt["ability"][norm(l[8:])] += 1
                elif l.startswith("- "): cnt["move"][norm(l[2:])] += 1
    return cnt


def main(out="data/entity_features.npz"):
    tmp = os.path.join(tempfile.gettempdir(), "entities.json")
    subprocess.run(["node", "tools/dump_entities.js", tmp], check=True)
    ents, vocab, sizes = json.load(open(tmp, encoding="utf-8")), json.load(open("data/vocab.json", encoding="utf-8")), load_vocab_sizes()
    counts, save = pool_counts(), {}
    for kind, size_key in KINDS.items():
        E, n = ents[kind], sizes[size_key]
        ids = np.array([vocab[kind][e["id"]] for e in E])
        parts = [CountVectorizer(binary=True, min_df=3, max_features=160, ngram_range=(1, 2), stop_words="english")
                 .fit_transform([e["desc"] for e in E]).toarray()]
        keys = collections.Counter(k for e in E for k in list(e["hooks"]) + list(e["flags"]))
        keys = [k for k, c in keys.most_common(40) if 3 <= c < len(E)]
        parts.append(np.array([[float(k in e["hooks"] or k in e["flags"]) for k in keys] for e in E]))
        X = np.hstack(parts).astype(np.float32)
        table = np.zeros((n, X.shape[1]), np.float32); table[ids] = X
        seen = np.zeros(n, np.int64)
        for e, i in zip(E, ids): seen[i] = counts[kind].get(norm(e["id"]), 0)
        seen[vocab[kind]["<unk>"]] = 1  # <unk>(novel)와 <unk>+1(hidden) 행은 실제로 학습되므로 0 처리 대상에서 제외
        seen[vocab[kind]["<unk>"] + 1] = 1
        save[f"{kind}_feat"], save[f"{kind}_seen"] = table, seen
        print(f"{kind}: {len(E)}개 개체, 특징 {X.shape[1]}d (BoW {parts[0].shape[1]} + 훅/플래그 {len(keys)}), 학습 노출 {int((seen > 0).sum())}개")
    np.savez_compressed(out, **save)
    print("저장:", out)


if __name__ == "__main__":
    main()
