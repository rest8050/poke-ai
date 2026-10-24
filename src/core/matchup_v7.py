"""매치업 특징 v7 (독립 구현, 상속 없음): matchup.py(v6)에 특성 EXEC 규칙(docs/ability_rules_design_A.md)을 더함. 파라미터 없음, 저장된 관측 텐서만 읽음.
v6/v4 모델이 matchup.py를 쓰므로 그쪽은 건드리지 않음 (v6 퇴역 시 matchup.py 삭제).

입력: my_cat/my_num/my_mv/opp_cat/opp_num/opp_mv (tensor_encoder 레이아웃), field_vec, 상대 특성 분포 opp_ab_prob [B,6,A], 상대 구애스카프 확률 opp_item_prob [B,6]
  num: 0 활성, 1 기절, 2 HP비율, 3~7 랭크/6 (공 방 특공 특방 스피드), 8 레벨, 9 테라 활성, 11~16 능력치/255 (내 쪽=실능력치, 상대=종족값), cat: 0 도구, 1 특성, 2~3 타입, 4 상태, 9 테라 타입
  mv: 0 위력/200, 3 우선도/7, 4 분류(0.33 물리 / 0.67 특수 / 1.0 변화), 6 타입ID/20 (안 드러난 기술칸은 전부 0)
  field_vec: 0 쾌청, 2 비, 10 트릭룸, 26/38 내/상대 순풍, 28~31 내 쪽 장판
규칙 (특성 id는 vocab, 상대는 확률 기대값):
  F_def[특성, 기술 타입] 방어측 배율 (부유류 면역 0, 두꺼운지방 불/얼음 0.5), 정밀기술 위력 <= 60 x1.5, 옹골참은 만피 KO 확률 제거,
  스피드: 랭크, 마비 x0.5, 스카프 x1.5, 순풍 x2, 날씨 특성(엽록소 쾌청 / 쓱쓱 비) x2, 트릭룸 반전 / 우선도: 장난꾸러기 변화기 +1 (악 타입 대상 실패),
  위협: 교체 후보가 위협이면 상대 활성의 공격/특공/스피드 배율 (상대 특성이 막거나 반응하는 결과 표), 재생력: 교체해 나갈 때 회복량
출력(dict):
  def  [B,6,24,4]  내 포켓몬 i가 상대 기술칸 j에 맞을 때: (배율/4, 데미지 비율[최대HP 대비, 1.5 상한 후 /1.5], 확정 KO 확률, 공격기 여부)
  off  [B,6,24,4]  상대 포켓몬 o가 내 기술칸 j에 맞을 때: 같은 4개
  spd_my [B,6] 내 포켓몬 i의 스피드 / 상대 활성 스피드 (log2, ±2 상한 후 /2), spd_opp [B,6] 상대 포켓몬 o / 내 활성 (둘 다 보정 스피드, 상대는 확률 기대값)
  fm_my [B,6,4] 내 포켓몬 i의 기술 j가 상대 활성보다 먼저일 기대값 (+1 먼저 / -1 나중 / 0 동률·불명·실패), fm_opp [B,6,4] 상대 o의 드러난 기술 j가 내 활성보다 먼저일 기대값
  entry_opp [B,6,3] 후보 i가 들어올 때 상대 활성의 (공격, 특공, 스피드) 배율 (후보가 위협이 아니면 1), entry_frac [B,6] 장판 진입 피해 비율, exit_heal [B] 내 활성이 나갈 때 회복량
근사(ponytail): 레벨 100 가정, 상대 능력치는 종족값 -> 252노력치/무보정 추정(2b+99, HP 2b+204; mult를 주면 신념이 예측한 배율), 랜덤 편차는 평균 0.925(KO 판정은 최댓값),
  급소/상태기 적용 가능성/흡수 이득/틀깨기는 미반영. 날씨는 쾌청/비 계열만, 우산·날씨 무효 특성 미반영"""
import json

import torch
import torch.nn as nn

TYPE_N = 21   # 0 없음, 1~18 타입, 19 stellar, 20 <unk> (vocab과 동일) — 0/19/20은 항상 등배
IMMUNE_ABILITIES = {"levitate": "ground", "eartheater": "ground", "flashfire": "fire", "wellbakedbody": "fire", "waterabsorb": "water",
                    "dryskin": "water", "stormdrain": "water", "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
                    "sapsipper": "grass"}
INTIM_DELTAS = (-1, 0, 1, 2)        # 위협 입장 시 상대 랭크 변화 후보
# 위협을 받은 상대 특성별 (공격, 특공, 스피드) 랭크 변화. 목록에 없으면 (-1, 0, 0). 소스: pokemon-showdown/data/abilities.ts
INTIM_RESULT = {"clearbody": (0, 0, 0), "whitesmoke": (0, 0, 0), "fullmetalbody": (0, 0, 0), "hypercutter": (0, 0, 0),
                "innerfocus": (0, 0, 0), "oblivious": (0, 0, 0), "owntempo": (0, 0, 0), "scrappy": (0, 0, 0), "mirrorarmor": (0, 0, 0),
                "guarddog": (1, 0, 0), "defiant": (1, 0, 0), "competitive": (-1, 2, 0), "rattled": (-1, 0, 1)}
WEATHER_SPEED = {"chlorophyll": 0, "swiftswim": 1}   # 스피드 x2가 되는 날씨 (0 쾌청, 1 비)
SUN, RAIN, TRICK_ROOM, TAILWIND_MY, TAILWIND_OPP = 0, 2, 10, 26, 38
COMBOS = [(w, s, p) for w in (0, 1) for s in (0, 1) for p in (0, 1)]    # 상대 불확실 요인(날씨 특성, 스카프, 장난꾸러기) 있다/없다 조합 8개


class MatchupV7(nn.Module):
    def __init__(self, vocab_path: str = "data/vocab.json", chart_path: str = "data/type_chart.json"):
        super().__init__()
        vocab = json.load(open(vocab_path, encoding="utf-8"))
        tid, ab, it = vocab["type"], vocab["ability"], vocab["item"]
        chart = torch.ones(TYPE_N, TYPE_N)
        for att, row in json.load(open(chart_path, encoding="utf-8")).items():
            for dfn, m in row.items():
                if att in tid and dfn in tid and tid[att] <= 18 and tid[dfn] <= 18:
                    chart[tid[att], tid[dfn]] = float(m)
        self.register_buffer("chart", chart, persistent=False)                     # [공격 타입, 방어 타입]
        n_ab = ab["<unk>"] + 2                                                     # 0..unk, 숨김(unk+1)까지 (신념 특성 로짓 폭과 같음)
        f_def = torch.ones(n_ab, TYPE_N)
        for name, t in IMMUNE_ABILITIES.items():
            f_def[ab[name], tid[t]] = 0.0
        f_def[ab["thickfat"], tid["fire"]] = f_def[ab["thickfat"], tid["ice"]] = 0.5
        self.register_buffer("f_def", f_def, persistent=False)
        flag = lambda name: torch.zeros(n_ab).index_fill_(0, torch.tensor([ab[name]]), 1.0)
        for k, name in (("is_tech", "technician"), ("is_sturdy", "sturdy"), ("is_prank", "prankster"), ("is_intim", "intimidate"), ("is_regen", "regenerator")):
            self.register_buffer(k, flag(name), persistent=False)
        weather = torch.zeros(n_ab, 2)                                             # 날씨 스피드 특성: 1이면 그 날씨에 스피드 x2
        for name, w in WEATHER_SPEED.items():
            weather[ab[name], w] = 1.0
        self.register_buffer("weather_speed", weather, persistent=False)
        intim = torch.zeros(n_ab, 3, len(INTIM_DELTAS))                            # [특성, (공격, 특공, 스피드), 변화 후보] 원핫
        default = (-1, 0, 0)
        for a in range(n_ab):
            for k, d in enumerate(default):
                intim[a, k, INTIM_DELTAS.index(d)] = 1.0
        for name, res in INTIM_RESULT.items():
            intim[ab[name]] = 0.0
            for k, d in enumerate(res):
                intim[ab[name], k, INTIM_DELTAS.index(d)] = 1.0
        self.register_buffer("intim_delta", intim, persistent=False)
        self.register_buffer("deltas", torch.tensor(INTIM_DELTAS, dtype=torch.float), persistent=False)
        self.balloon, self.scarf = it.get("airballoon", -1), it["choicescarf"]
        self.burn, self.par = vocab["status"]["brn"], vocab["status"]["par"]
        self.ground, self.dark, self.flying, self.rock = tid["ground"], tid["dark"], tid["flying"], tid["rock"]
        self.unaware, self.levitate, self.magicguard, self.boots = ab["unaware"], ab["levitate"], ab["magicguard"], it["heavydutyboots"]
        self.register_buffer("spikes_frac", torch.tensor([0.0, 1 / 8, 1 / 6, 1 / 4]), persistent=False)
        self.register_buffer("combo_bits", torch.tensor(COMBOS, dtype=torch.float), persistent=False)       # [8,3]

    @staticmethod
    def _stage(r):  # 랭크/6 → 배율
        s = (r * 6).round()
        return torch.where(s >= 0, (2 + s) / 2, 2 / (2 - s))

    def _stats(self, num, mine: bool, mult=None, staged: bool = True):
        v = num[..., 11:17] * 255.0                                                # hp, atk, def, spa, spd, spe
        if not mine:
            v = torch.cat([2 * v[..., :1] + 204, 2 * v[..., 1:] + 99], -1)
            if mult is not None:                                                   # 신념이 예측한 노력치/성격 배율 [..., 6] (hp, atk, def, spa, spd, spe)
                v = v * mult
        st = self._stage(num[..., 3:8]) if staged else torch.ones_like(num[..., 3:8])    # atk, def, spa, spd, spe (staged=False: 랭크 무시 = 천진 상대)
        return v[..., 0], v[..., 1] * st[..., 0], v[..., 2] * st[..., 1], v[..., 3] * st[..., 2], v[..., 4] * st[..., 3], v[..., 5] * st[..., 4]

    def _onehot(self, cat):
        """내 쪽 특성 id -> 원핫 [B,6,A]"""
        return torch.nn.functional.one_hot(cat[..., 1].clamp(0, self.f_def.size(0) - 1), self.f_def.size(0)).float()

    def entry_frac(self, cat, num, field_vec):
        """교체해 들어올 때 내 쪽 장판에 깎이는 HP 비율 [B,6] (활성은 0). 스텔스록 1/8 x 바위 상성 + 압정 1/8,1/6,1/4(접지만), 두꺼운부츠/매직가드는 면제"""
        tera = num[..., 9] > 0.5
        t1 = torch.where(tera, cat[..., 9], cat[..., 2]).clamp(0, TYPE_N - 1)
        t2 = torch.where(tera, torch.zeros_like(cat[..., 3]), cat[..., 3]).clamp(0, TYPE_N - 1)
        ab, it = cat[..., 1], cat[..., 0]
        rock = self.chart[self.rock, t1] * self.chart[self.rock, t2]
        grounded = (t1 != self.flying) & (t2 != self.flying) & (ab != self.levitate) & (it != self.balloon)
        spikes = self.spikes_frac[(field_vec[:, 29] * 3).round().long().clamp(0, 3)][:, None]
        frac = 0.125 * rock * (field_vec[:, 28:29] > 0.5) + spikes * grounded
        safe = (it == self.boots) | (ab == self.magicguard) | (num[..., 0] > 0.5)
        return torch.where(safe, torch.zeros_like(frac), frac)

    def _entry_opp(self, my_cat, my_num, opp_num, q_opp):
        """교체 후보 i가 위협이면 들어올 때 상대 활성의 (공격, 특공, 스피드) 배율 [B,6,3]. 상대 특성은 분포 q_opp [B,6,A]의 기대값"""
        B = my_cat.size(0)
        ar = torch.arange(B, device=my_cat.device)
        oa = opp_num[..., 0].argmax(1)
        q_act = q_opp[ar, oa]                                                      # [B,A]
        cur = (opp_num[ar, oa][:, [3, 5, 7]] * 6).round()                           # 상대 활성의 (공격, 특공, 스피드) 현재 랭크 [B,3]
        new = (cur[..., None] + self.deltas).clamp(-6, 6)                           # [B,3,4]
        ratio = self._stage(new / 6) / self._stage(cur / 6)[..., None]              # [B,3,4]
        p = torch.einsum("ba,akd->bkd", q_act, self.intim_delta)                    # [B,3,4] 변화 후보별 확률
        factor = (p * ratio).sum(-1)                                                # [B,3]
        cand = (self.is_intim[my_cat[..., 1].clamp(0, self.is_intim.size(0) - 1)] > 0) & (my_num[..., 0] < 0.5) & (my_num[..., 11] > 0) & (my_num[..., 1] < 0.5)
        return torch.where(cand[..., None], factor[:, None, :], torch.ones(B, 6, 3, device=my_cat.device))

    def _pair(self, att_cat, att_num, att_mv, att_mine, q_att, def_cat, def_num, def_mine, q_def, att_mult=None, def_mult=None, entry=None, cand_f=None):
        """공격측 6마리(마리당 기술 K칸, 보통 4) x 방어측 6마리 → [B, 6(방어), 6K(공격 기술칸), 4]. q_*: 특성 분포 [B,6,A].
        cand_f: (공격 배율, 특공 배율) [B,6] — 방어측 후보 i가 위협으로 들어올 때 공격측 활성의 능력치 배율 (활성이 아닌 공격측 포켓몬에는 적용 안 함)"""
        B, K = att_cat.size(0), att_mv.size(2)
        a_hp, a_atk, _, a_spa, _, _ = self._stats(att_num, att_mine, att_mult)
        d_hp, _, d_def, _, d_spd, _ = self._stats(def_num, def_mine, def_mult)
        _, a_atk_raw, _, a_spa_raw, _, _ = self._stats(att_num, att_mine, att_mult, staged=False)       # 천진: 랭크 무시 계산용
        _, _, d_def_raw, _, d_spd_raw, _ = self._stats(def_num, def_mine, def_mult, staged=False)
        rep = lambda x: x.repeat_interleave(K, dim=1)[:, None]                     # [B,6] → [B,1,6K]
        mv = att_mv.reshape(B, 6 * K, -1)
        bp, kind, mt = mv[..., 0] * 200.0, mv[..., 4], (mv[..., 6] * 20.0).round().long().clamp(0, TYPE_N - 1)
        a_present = (att_num[..., 11] > 0) & (att_num[..., 1] < 0.5)
        revealed = (att_mv.abs().sum(-1).reshape(B, 6 * K) > 0) & rep(a_present.float())[:, 0].bool()
        damaging = revealed & (bp > 0) & (kind < 0.85)                             # [B,24]
        phys = (kind < 0.5)[:, None]
        # 공격측: 자속/테라/화상/정밀기술
        t1, t2, tera = rep(att_cat[..., 2]), rep(att_cat[..., 3]), rep(att_cat[..., 9])
        tera_on = rep((att_num[..., 9] > 0.5).float()).bool()
        mt_ = mt[:, None]
        stab = ((mt_ == t1) | (mt_ == t2) | ((mt_ == tera) & tera_on)).float() * 0.5 + 1.0
        burned = rep((att_cat[..., 4] == self.burn).float()) * phys.float()
        p_tech = rep(torch.einsum("bia,a->bi", q_att, self.is_tech))               # [B,1,6K]
        bp_eff = bp[:, None] * (1.0 + 0.5 * p_tech * (bp[:, None].round() <= 60).float())      # 위력은 정수라 부동소수 오차는 반올림으로 흡수
        A = torch.where(phys, rep(a_atk), rep(a_spa))                              # [B,1,6K]
        A = torch.where((def_cat[..., 1] == self.unaware)[..., None], torch.where(phys, rep(a_atk_raw), rep(a_spa_raw)), A)     # [B,6,6K]
        if cand_f is not None:
            f = torch.where(phys, cand_f[0][:, :, None], cand_f[1][:, :, None])    # [B,6,6K]
            A = A * torch.where(rep((att_num[..., 0] > 0.5).float()).bool(), f, torch.ones_like(f))
        # 방어측: 타입(테라면 테라 타입 하나)/능력치/특성 배율 (면역, 두꺼운지방)
        d_t1 = torch.where(def_num[..., 9] > 0.5, def_cat[..., 9], def_cat[..., 2])[..., None]   # [B,6,1]
        d_t2 = torch.where(def_num[..., 9] > 0.5, torch.zeros_like(def_cat[..., 3]), def_cat[..., 3])[..., None]
        eff = self.chart[mt_.expand(-1, 6, -1), d_t1.clamp(0, TYPE_N - 1).expand(-1, -1, 6 * K)] * \
            self.chart[mt_.expand(-1, 6, -1), d_t2.clamp(0, TYPE_N - 1).expand(-1, -1, 6 * K)]
        if self.balloon >= 0:
            eff = torch.where((def_cat[..., 0] == self.balloon)[..., None] & (mt_ == self.ground), torch.zeros_like(eff), eff)
        by_type = torch.einsum("bia,at->bit", q_def, self.f_def)                    # [B,6,T] 방어측 특성의 타입별 배율 기대값
        eff = eff * by_type.gather(2, mt_.expand(-1, 6, -1))
        D = torch.where(phys, d_def[..., None], d_spd[..., None])                  # [B,6,1] -> (천진 공격측이면) [B,6,24]
        D = torch.where(rep((att_cat[..., 1] == self.unaware).float()).bool(), torch.where(phys, d_def_raw[..., None], d_spd_raw[..., None]), D)
        D = D.clamp_min(1.0)
        base = (42.0 * bp_eff * A / D) / 50.0 + 2.0
        dmg = base * stab * eff * (1.0 - 0.5 * burned)                             # 최댓값(랜덤 1.0) 기준
        hp = d_hp[..., None].clamp_min(1.0)
        cur = hp * (def_num[..., 2:3] if entry is None else (def_num[..., 2:3] - entry[..., None]).clamp_min(0.0))      # 장판에 깎인 뒤 HP
        dm = damaging[:, None]
        frac = torch.where(dm, dmg * 0.925 / hp, torch.zeros_like(dmg))
        p_surv = torch.einsum("bia,a->bi", q_def, self.is_sturdy)[..., None] * (def_num[..., 2:3] >= 0.999).float()     # 옹골참: 만피에서 한 방은 버팀
        ko = (dm & (dmg >= cur)).float() * (1.0 - p_surv)
        d_ok = ((def_num[..., 11] > 0) & (def_num[..., 1] < 0.5))[..., None]       # 방어측이 없거나 기절이면 0
        out = torch.stack([torch.where(dm, eff / 4.0, torch.zeros_like(eff)), frac.clamp(max=1.5) / 1.5, ko, dm.expand_as(eff).float()], -1)
        return out * d_ok[..., None].float()

    # ---- 스피드 / 선공 (`_speed`) ----

    def forward(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, field_vec, opp_ab_prob, opp_item_prob, opp_mult=None):
        B, dev = my_cat.size(0), my_cat.device
        ar = torch.arange(B, device=dev)
        q_my = self._onehot(my_cat)
        q_opp = opp_ab_prob
        ent = self.entry_frac(my_cat, my_num, field_vec)
        entry_opp = self._entry_opp(my_cat, my_num, opp_num, q_opp)
        D = self._pair(opp_cat, opp_num, opp_mv, False, q_opp, my_cat, my_num, True, q_my, att_mult=opp_mult, entry=ent, cand_f=(entry_opp[..., 0], entry_opp[..., 1]))      # 내 포켓몬이 맞음
        O = self._pair(my_cat, my_num, my_mv, True, q_my, opp_cat, opp_num, False, q_opp, def_mult=opp_mult)                                                         # 상대 포켓몬이 맞음
        my_act, opp_act = my_num[..., 0].argmax(1), opp_num[..., 0].argmax(1)
        out = {"def": D, "off": O, "entry_opp": entry_opp, "entry_frac": ent}
        out.update(self._speed(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, field_vec, q_my, q_opp, opp_item_prob, opp_mult, my_act, opp_act))
        hp_act = my_num[ar, my_act, 2]
        out["exit_heal"] = (q_my[ar, my_act] @ self.is_regen) * torch.clamp(1 - hp_act, max=1 / 3)
        return out

    def _speed(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, field_vec, q_my, q_opp, p_scarf, opp_mult, my_act, opp_act):
        B, dev = my_cat.size(0), my_cat.device
        ar = torch.arange(B, device=dev)
        sun, rain = field_vec[:, SUN] > 0.5, field_vec[:, RAIN] > 0.5
        tr = (field_vec[:, TRICK_ROOM] > 0.5).float()
        par_f = lambda cat: torch.where(cat[..., 4] == self.par, 0.5, 1.0)
        # 날씨 특성 배율: 내 쪽은 확정(1 또는 2), 상대는 2가 될 확률
        w_my = torch.einsum("bia,aw->biw", q_my, self.weather_speed)               # [B,6,2] 0/1
        w_opp = torch.einsum("bia,aw->biw", q_opp, self.weather_speed)             # [B,6,2] 확률
        pick = lambda w: torch.where(sun[:, None], w[..., 0], torch.where(rain[:, None], w[..., 1], torch.zeros_like(w[..., 0])))
        pw_my, pw_opp = pick(w_my), pick(w_opp)
        spe_my = self._stats(my_num, True)[5] * par_f(my_cat) * torch.where(field_vec[:, TAILWIND_MY, None] > 0.5, 2.0, 1.0)
        spe_my = spe_my * (1.0 + pw_my) * torch.where(my_cat[..., 0] == self.scarf, 1.5, 1.0)                  # 확정이라 기대값이 곧 값
        spe_opp = self._stats(opp_num, False, opp_mult)[5] * par_f(opp_cat) * torch.where(field_vec[:, TAILWIND_OPP, None] > 0.5, 2.0, 1.0)     # 불확실 배율 제외
        # 상대 조합 8개: (날씨 특성, 스카프, 장난꾸러기) 있음/없음
        bits = self.combo_bits                                                      # [8,3]
        pu = torch.stack([pw_opp, p_scarf, torch.einsum("bia,a->bi", q_opp, self.is_prank)], -1)         # [B,6,3] 상대 각 마리의 불확실 요인 확률
        wgt = torch.prod(torch.where(bits[None, None] > 0.5, pu[:, :, None, :], 1 - pu[:, :, None, :]), -1)   # [B,6,8]
        spe_opp_c = spe_opp[..., None] * (1.0 + bits[:, 0]) * (1.0 + 0.5 * bits[:, 1])                     # [B,6,8]
        # 우선도
        pri = lambda mv: (mv[..., 3] * 7).round()                                    # [B,6,4]
        is_status = lambda mv: mv[..., 4] > 0.85
        revealed = lambda mv, num: (mv.abs().sum(-1) > 0) & (num[..., 11:12] > 0)
        my_pri, my_st, my_rev = pri(my_mv), is_status(my_mv), revealed(my_mv, my_num)
        op_pri, op_st, op_rev = pri(opp_mv), is_status(opp_mv), revealed(opp_mv, opp_num)
        prank_my = torch.einsum("bia,a->bi", q_my, self.is_prank)
        my_pri_eff = my_pri + prank_my[..., None] * my_st.float()                                  # [B,6,4] 확정
        # 상쇄 대상: 상대 활성의 드러난 기술 최고 우선도 (조합별로 장난꾸러기 +1), 내 활성의 최고 우선도
        NEG = -100.0
        oa_pri, oa_st, oa_rev = op_pri[ar, opp_act], op_st[ar, opp_act], op_rev[ar, opp_act]                  # [B,4]
        oa_best_c = torch.stack([torch.where(oa_rev, oa_pri + bits[c, 2] * oa_st.float(), torch.full_like(oa_pri, NEG)).amax(-1) for c in range(8)], -1)   # [B,8]
        oa_best_c = torch.where(oa_best_c <= NEG / 2, torch.zeros_like(oa_best_c), oa_best_c)
        ma_best = torch.where(my_rev[ar, my_act], my_pri_eff[ar, my_act], torch.full_like(my_pri[:, 0], NEG)).amax(-1)       # [B]
        ma_best = torch.where(ma_best <= NEG / 2, torch.zeros_like(ma_best), ma_best)
        # 타입: 악 타입 대상은 장난꾸러기 변화기 실패
        def is_dark(cat, num, act):
            tera = num[ar, act, 9] > 0.5
            t1 = torch.where(tera, cat[ar, act, 9], cat[ar, act, 2]); t2 = torch.where(tera, torch.zeros_like(cat[ar, act, 3]), cat[ar, act, 3])
            return (t1 == self.dark) | (t2 == self.dark)
        dark_opp_act, dark_my_act = is_dark(opp_cat, opp_num, opp_act), is_dark(my_cat, my_num, my_act)
        sgn = lambda x: torch.sign(x)
        spe_oa_c = spe_opp_c[ar, opp_act]                                           # [B,8]
        w_oa = wgt[ar, opp_act]                                                     # [B,8]
        # (1) 내 기술칸 vs 상대 활성
        cmp_p = sgn(my_pri_eff[..., None] - oa_best_c[:, None, None, :])             # [B,6,4,8]
        cmp_s = sgn(spe_my[:, :, None, None] - spe_oa_c[:, None, None, :]) * (1 - 2 * tr)[:, None, None, None]
        s1 = torch.where(cmp_p != 0, cmp_p, cmp_s)
        fm_my = (s1 * w_oa[:, None, None, :]).sum(-1)
        fail = (prank_my[..., None] > 0) & my_st & dark_opp_act[:, None, None]
        fm_my = torch.where(my_rev & ~fail, fm_my, torch.zeros_like(fm_my)) * (my_num[..., 11:12] > 0).float()
        # (2) 상대 드러난 기술칸 vs 내 활성 (상대 각 마리의 조합 8개)
        op_pri_c = op_pri[..., None] + bits[:, 2] * op_st.float()[..., None]        # [B,6,4,8]
        cmp_p = sgn(op_pri_c - ma_best[:, None, None, None])
        cmp_s = sgn(spe_opp_c[:, :, None, :] - spe_my[ar, my_act][:, None, None, None]) * (1 - 2 * tr)[:, None, None, None]
        s2 = torch.where(cmp_p != 0, cmp_p, cmp_s)
        fail = (bits[:, 2] > 0.5)[None, None, None, :] & op_st[..., None] & dark_my_act[:, None, None, None]
        s2 = torch.where(fail, torch.zeros_like(s2), s2)
        fm_opp = (s2 * wgt[:, :, None, :]).sum(-1)
        fm_opp = torch.where(op_rev, fm_opp, torch.zeros_like(fm_opp))
        r = lambda a, b: (torch.log2(a.clamp_min(1.0) / b.clamp_min(1.0)).clamp(-2, 2) / 2)
        spd_my = (r(spe_my[..., None], spe_oa_c[:, None, :]) * w_oa[:, None, :]).sum(-1)               # [B,6]
        spd_opp = (r(spe_opp_c, spe_my[ar, my_act][:, None, None]) * wgt).sum(-1)                      # [B,6]
        return {"spd_my": spd_my, "spd_opp": spd_opp, "fm_my": fm_my, "fm_opp": fm_opp}
