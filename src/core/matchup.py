"""매치업 특징: 이미 있는 관측 텐서만으로 "상대 기술이 내 포켓몬에게 얼마나 아픈가 / 내 기술이 상대 포켓몬에게 얼마나 아픈가"를 계산.
같은 함수를 학습과 서버 추론이 모두 쓰므로 데이터 재빌드가 필요 없고 두 쪽이 어긋나지 않음 (인코더가 이미 넣는 값만 사용).

입력: my_cat/my_num/my_mv/opp_cat/opp_num/opp_mv (tensor_encoder 레이아웃)
  num: 0 활성, 1 기절, 2 HP비율, 3~7 랭크/6, 8 레벨, 9 테라 활성, 11~16 능력치/255 (내 쪽=실능력치, 상대=종족값), cat: 1 특성, 2~3 타입, 4 상태, 5 도구는 0번, 9 테라 타입
  mv: 0 위력/200, 4 분류(0.33 물리 / 0.67 특수 / 1.0 변화), 6 타입ID/20 (안 드러난 기술칸은 전부 0)
출력(dict):
  def  [B,6,24,4]  내 포켓몬 i가 상대 기술칸 j(상대 포켓몬 j//4의 j%4번 기술)에 맞을 때: (배율/4, 데미지 비율[최대HP 대비, 1.5 상한 후 /1.5], 확정 KO 가능, 공격기 여부)
  off  [B,6,24,4]  상대 포켓몬 o가 내 기술칸 j(내 포켓몬 j//4의 j%4번 기술)에 맞을 때: 같은 4개
  spd_my [B,6] 내 포켓몬 i의 스피드 / 상대 활성 스피드 (log2, ±2 상한 후 /2), spd_opp [B,6] 상대 포켓몬 o / 내 활성
근사(ponytail): 레벨 100 가정, 상대 능력치는 종족값 → 252노력치/무보정 추정(2b+99, HP 2b+204; mult를 주면 신념이 예측한 배율을 곱함), 랜덤 편차는 평균 0.925(KO 판정은 최댓값 1.0),
  급소/도구/특성 위력 보정/벽/공중부양 외 특성 면역은 미반영. 테라: 방어는 테라 타입 하나, 공격 자속은 원래 타입 + 테라 타입"""
import json

import torch
import torch.nn as nn

TYPE_N = 21  # 0 없음, 1~18 타입, 19 stellar, 20 <unk> (vocab과 동일) — 0/19/20은 항상 등배
IMMUNE_ABILITIES = {"levitate": "ground", "eartheater": "ground", "flashfire": "fire", "wellbakedbody": "fire", "waterabsorb": "water",
                    "dryskin": "water", "stormdrain": "water", "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
                    "sapsipper": "grass"}


class Matchup(nn.Module):
    def __init__(self, vocab_path: str = "data/vocab.json", chart_path: str = "data/type_chart.json"):
        super().__init__()
        vocab = json.load(open(vocab_path, encoding="utf-8"))
        tid = vocab["type"]
        chart = torch.ones(TYPE_N, TYPE_N)
        raw = json.load(open(chart_path, encoding="utf-8"))
        for att, row in raw.items():
            for dfn, m in row.items():
                if att in tid and dfn in tid and tid[att] <= 18 and tid[dfn] <= 18:
                    chart[tid[att], tid[dfn]] = float(m)
        self.register_buffer("chart", chart, persistent=False)                     # [공격 타입, 방어 타입]
        n_ab = max(vocab["ability"].values()) + 3                                  # <unk>, 숨김(unk+1) 포함
        imm = torch.zeros(n_ab, dtype=torch.long)
        for name, t in IMMUNE_ABILITIES.items():
            if name in vocab["ability"]:
                imm[vocab["ability"][name]] = tid[t]
        self.register_buffer("ab_immune", imm, persistent=False)
        self.balloon = vocab["item"].get("airballoon", -1)
        self.burn = vocab["status"]["brn"]
        self.ground = tid["ground"]

    @staticmethod
    def _stage(r):  # 랭크/6 → 배율
        s = (r * 6).round()
        return torch.where(s >= 0, (2 + s) / 2, 2 / (2 - s))

    def _stats(self, num, mine: bool, mult=None):
        v = num[..., 11:17] * 255.0                                                # hp, atk, def, spa, spd, spe
        if not mine:
            v = torch.cat([2 * v[..., :1] + 204, 2 * v[..., 1:] + 99], -1)
            if mult is not None:                                                   # 신념이 예측한 노력치/성격 배율 [..., 6] (hp, atk, def, spa, spd, spe)
                v = v * mult
        st = self._stage(num[..., 3:8])                                            # atk, def, spa, spd, spe
        return v[..., 0], v[..., 1] * st[..., 0], v[..., 2] * st[..., 1], v[..., 3] * st[..., 2], v[..., 4] * st[..., 3], v[..., 5] * st[..., 4]

    def _pair(self, att_cat, att_num, att_mv, att_mine, def_cat, def_num, def_mine, att_mult=None, def_mult=None):
        """공격측 6마리(마리당 기술 K칸, 보통 4) x 방어측 6마리 → [B, 6(방어), 6K(공격 기술칸), 4]. mult: 상대 쪽 능력치 배율(기본 없음 = 표준 가정)"""
        B, K = att_cat.size(0), att_mv.size(2)
        a_hp, a_atk, _, a_spa, _, _ = self._stats(att_num, att_mine, att_mult)
        d_hp, _, d_def, _, d_spd, _ = self._stats(def_num, def_mine, def_mult)
        rep = lambda x: x.repeat_interleave(K, dim=1)[:, None]                     # [B,6] → [B,1,6K]
        mv = att_mv.reshape(B, 6 * K, -1)
        bp, kind, mt = mv[..., 0] * 200.0, mv[..., 4], (mv[..., 6] * 20.0).round().long().clamp(0, TYPE_N - 1)
        a_present = (att_num[..., 11] > 0) & (att_num[..., 1] < 0.5)
        revealed = (att_mv.abs().sum(-1).reshape(B, 6 * K) > 0) & rep(a_present.float())[:, 0].bool()
        damaging = revealed & (bp > 0) & (kind < 0.85)                             # [B,24]
        phys = (kind < 0.5)[:, None]
        # 공격측: 자속/테라/화상
        t1, t2, tera = rep(att_cat[..., 2]), rep(att_cat[..., 3]), rep(att_cat[..., 9])
        tera_on = rep((att_num[..., 9] > 0.5).float()).bool()
        mt_ = mt[:, None]
        stab = ((mt_ == t1) | (mt_ == t2) | ((mt_ == tera) & tera_on)).float() * 0.5 + 1.0
        burned = rep((att_cat[..., 4] == self.burn).float()) * phys.float()
        A = torch.where(phys, rep(a_atk), rep(a_spa))                              # [B,1,24]
        # 방어측: 타입(테라면 테라 타입 하나)/능력치/특성 면역
        d_t1 = torch.where(def_num[..., 9] > 0.5, def_cat[..., 9], def_cat[..., 2])[..., None]   # [B,6,1]
        d_t2 = torch.where(def_num[..., 9] > 0.5, torch.zeros_like(def_cat[..., 3]), def_cat[..., 3])[..., None]
        eff = self.chart[mt_.expand(-1, 6, -1), d_t1.clamp(0, TYPE_N - 1).expand(-1, -1, 6 * K)] * \
            self.chart[mt_.expand(-1, 6, -1), d_t2.clamp(0, TYPE_N - 1).expand(-1, -1, 6 * K)]
        imm_t = self.ab_immune[def_cat[..., 1].clamp(0, len(self.ab_immune) - 1)][..., None]    # [B,6,1]
        immune = (imm_t == mt_) & (imm_t > 0)
        if self.balloon >= 0:
            immune = immune | ((def_cat[..., 0] == self.balloon)[..., None] & (mt_ == self.ground))
        eff = torch.where(immune, torch.zeros_like(eff), eff)
        D = torch.where(phys, d_def[..., None], d_spd[..., None]).clamp_min(1.0)  # [B,6,1]
        base = (42.0 * bp[:, None] * A / D) / 50.0 + 2.0
        dmg = base * stab * eff * (1.0 - 0.5 * burned)                             # 최댓값(랜덤 1.0) 기준
        hp = d_hp[..., None].clamp_min(1.0)
        cur = hp * def_num[..., 2:3]
        dm = damaging[:, None]
        frac = torch.where(dm, dmg * 0.925 / hp, torch.zeros_like(dmg))
        ko = (dm & (dmg >= cur)).float()
        d_ok = ((def_num[..., 11] > 0) & (def_num[..., 1] < 0.5))[..., None]       # 방어측이 없거나 기절이면 0
        out = torch.stack([torch.where(dm, eff / 4.0, torch.zeros_like(eff)), frac.clamp(max=1.5) / 1.5, ko, dm.expand_as(eff).float()], -1)
        return out * d_ok[..., None].float()

    def forward(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, opp_mult=None):
        D = self._pair(opp_cat, opp_num, opp_mv, False, my_cat, my_num, True, att_mult=opp_mult)      # 내 포켓몬이 맞음
        O = self._pair(my_cat, my_num, my_mv, True, opp_cat, opp_num, False, def_mult=opp_mult)       # 상대 포켓몬이 맞음
        ar = torch.arange(my_num.size(0), device=my_num.device)
        spe_my = self._stats(my_num, True)[5]
        spe_opp = self._stats(opp_num, False, opp_mult)[5]
        my_act, opp_act = my_num[..., 0].argmax(1), opp_num[..., 0].argmax(1)
        r = lambda a, b: (torch.log2(a.clamp_min(1.0) / b.clamp_min(1.0)).clamp(-2, 2) / 2)
        return {"def": D, "off": O, "spd_my": r(spe_my, spe_opp[ar, opp_act][:, None]), "spd_opp": r(spe_opp, spe_my[ar, my_act][:, None])}
