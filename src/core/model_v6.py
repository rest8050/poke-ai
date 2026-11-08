"""현행 정책/가치 네트워크 v6 (독립 구조, 상속 없음): 엔티티 토큰 트랜스포머 + 규칙 기반 매치업 + 세트 추정 + 합법성/효과 입력.
1) 규모: d_model 256 / 6층 / 8헤드. 한 턴 = 포켓몬 12 + 기술 48 + 필드 3 + CLS 1 = 64개 토큰, 압축 없이 전부 어텐션
2) 매치업 특징(src/core/matchup.py, 파라미터 없음): 상대 기술이 내 포켓몬에 주는 배율·데미지·KO, 내 기술이 상대 포켓몬에 주는 효과, 스피드 우열
   - 포켓몬/기술 토큰 입력에 요약치를 더하고, 어텐션 로짓에 편향 bias[h,i,j] = f_h(배율, 데미지, KO, 공격기여부)를 더함 (층마다 따로 학습)
   - 교체 후보 점수와 교체/기술 판단(type_head)에 후보별·활성 매치업 요약을 직접 입력
   - 천진은 상대 공격/방어 랭크를 무시하고 계산, 교체 후보의 데미지/KO는 장판(스텔스록·압정) 피해를 뺀 HP 기준, 후보별 진입 피해 비율을 후보 특징과 포켓몬 토큰에 추가
3) 상대 세트 추정(belief, src/core/belief.py): 자기 완결형 모듈, 출력은 detach라서 정책/가치 손실은 세트 추정에 닿지 않음 (숨김정보 손실로만 학습)
   - 숨은 기술은 마리당 확률 상위 12개 후보의 혼합으로 취급: 후보의 세트 포함 확률 w(= 미공개 칸 수 x 확률, 최대 1)로 포켓몬 쌍마다
     P_KO = 1 - Π(1 - w·KO), 그럴듯한 최대 데미지 = w >= 0.15인 후보의 최대 데미지 → 포켓몬 토큰 요약, 교체 후보 특징, 포켓몬<->포켓몬 어텐션 편향에 반영
   - 세트 추정이 예측한 상대 능력치 배율(노력치/성격)을 매치업의 상대 능력치에 곱하고 상대 포켓몬 토큰에도 입력
4) 합법성 입력: 서버 요청 기반 순수 합법 마스크(legal_mask, 휴리스틱 가지치기 없음)를 특징으로 받음 (구애·돌격조끼 고정, 도발·앵콜·사슬묶기, PP 소진, 교체 봉쇄)
   - 토큰: 내 활성 기술 토큰마다 "사용 가능" 플래그, 내 포켓몬 토큰마다 "교체 가능" 플래그
   - 요약(LEG_G=25): 합법 비트 14 + 합법 기술 수·공격기 수·교체 수·테라 가능·"기술 한 개만 가능"·"고정 의심" 6 + 사용 가능한 기술만으로 다시 계산한 공격 요약 3 + 고정으로 잃은 화력 2
   - legal_mask를 안 주면 action_mask를 씀 (학습 데이터는 둘이 같음). 배포에서는 legal_mask(합법성만)를 특징으로, action_mask(가지치기 포함)를 로짓 마스킹으로 따로 넘김
5) 행동 제약 효과: 양쪽 활성 포켓몬의 도발/앵콜/사슬묶기 + 교체 봉쇄 플래그와 남은 턴 비율(field_vec 뒤쪽 14열). 필드 토큰과 턴 벡터(히스토리)로 들어가고
   합법성 요약과 함께 융합층/종류 헤드/교체 점수에도 직접 입력. 옛 데이터의 짧은 field_vec은 0으로 채움
6) 히스토리: 턴 요약 = [CLS, 내 활성 토큰, 상대 활성 토큰, 필드/직전 사건 벡터]을 인과 트랜스포머로 요약
7) 미공개 기술 칸도 토큰으로 유지 (빈 슬롯만 마스킹) + 포켓몬 토큰에 공개된 기술 수/4
8) my_pos=False: 내 쪽 팀 슬롯/기술 칸 위치 임베딩을 끔 → 내 팀 순서 순열에 대해 정확히 대칭 (순서가 팀 지문 통로가 되는 것을 막음). 상대 쪽은 공개된 순서라 유지
인터페이스: forward/forward_sequences/get_action, cfg + save_ckpt/model_from_ckpt.
arch: {"model": "v6", "my_pos": false}"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.belief import STAT_SCALE, SetBelief, build_move_table
from src.core.matchup import Matchup
from src.core.model import MAX_TURNS, PokemonEmbeddingLayer, get_action
from src.core.tensor_encoder import EFFECT_DIM, EFFECT_START, EXT_START, MOVE_NUM_DIM

# 3차원 어텐션 마스크(헤드별 편향)는 추론용 빠른 커널에서 학습 경로와 값이 달라짐(CLS 기준 2e-3) → 학습과 같은 일반 경로만 쓰도록 끔
torch.backends.mha.set_fastpath_enabled(False)

N_TOK = 64            # 포켓몬 12 + 기술 48 + 필드 3 + CLS 1
MV_EXTRA = 3
OLD_NUM = 27          # v6가 학습한 포켓몬별 수치 폭 (인코더가 이후에 추가한 열은 버림: 옛 체크포인트 호환)
BASE_IN = 177 + OLD_NUM         # 포켓몬 토큰 기본 입력 차원 (종족값 투영 64 + 도구/특성/타입/상태/테라 임베딩 + 배틀 수치 + 이벤트 증거)
K_GUESS = 12          # 마리당 후보 기술 수 (숨은 기술 재현율: 상위 4 0.73 / 상위 8 0.94 / 상위 12 0.98)
PLAUSIBLE = 0.15      # 이 이상의 세트 포함 확률이면 "그럴듯한" 후보
MULT_RANGE = (0.6, 1.25)
LEG_G = 14 + 6 + 3 + 2
MON_X = 7 + 2 + 6 + 1    # 포켓몬 토큰 추가 수치: 매치업 요약 7 + 숨은 기술 위협 2 + 상대 능력치 예측 6 + 진입 피해 비율 1
CAND_N = 9 + 2 + 1       # 교체 후보 특징: 요약 9 + 상대 활성의 숨은 기술 위협 2 + 진입 피해 비율 1
G_DIM = LEG_G + EFFECT_DIM   # 전역 요약: 합법성 요약 + 양쪽 활성의 효과 열


def _mx(x):           # [..., 4(기술), 4(채널)] 중 앞 3채널(배율, 데미지, KO)의 기술별 최댓값 → [..., 3]
    return x[..., :3].amax(-2)


class EntityPokemonNetV6(nn.Module):
    ACTION_DIM = 22
    uses_legal = True

    def __init__(self, vocab_path: str = "data/vocab.json", my_pos: bool = True, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0, history_dim: int = 256, hist_layers: int = 2, latent_dim: int = 256,
                 head_width: int = 192, entity_features: bool = True,
                 species_dim: int = 32, belief_hidden: int = 256, belief_species_dropout: float = 0.3):
        super().__init__()
        self.cfg = dict(model="v6", d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                        history_dim=history_dim, hist_layers=hist_layers, latent_dim=latent_dim, head_width=head_width,
                        species_dim=species_dim, belief_hidden=belief_hidden, belief_species_dropout=belief_species_dropout, my_pos=my_pos)
        d = self.d = d_model
        field_dim = EXT_START      # v6가 학습한 field_vec 폭 (뒤에 붙은 확장 블록은 _fit_field가 버림)
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        pos = torch.ones(12) if my_pos else torch.tensor([0.0] * 6 + [1.0] * 6)       # 내 쪽 슬롯/기술 칸 순서는 임의(팀 export 순서)라서 끌 수 있음. 상대 쪽 순서는 공개된 순서라 정보가 있어 유지
        self.register_buffer("pos_gate", pos, persistent=False)
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        self.matchup = Matchup(vocab_path, entry=True)
        self.belief = SetBelief(vocab_path, species_dim, belief_hidden, belief_species_dropout, num_dim=OLD_NUM)
        self.register_buffer("move_table", build_move_table(vocab_path), persistent=False)
        self.unk_move = nn.Parameter(torch.zeros(d))                        # 미공개 기술 칸 표시
        self.hid_proj = nn.Linear(1 + MOVE_NUM_DIM + 48, d)                 # 후보 기술 혼합(확률 합, 수치, 임베딩) -> 상대 포켓몬 토큰
        self.mon_proj = nn.Sequential(nn.Linear(BASE_IN + MON_X + 1 + 1, d), nn.LayerNorm(d))      # + 공개된 기술 수 + 교체 가능 플래그
        self.move_proj = nn.Sequential(nn.Linear(142 + MV_EXTRA + 1, d), nn.LayerNorm(d))          # + 사용 가능 플래그
        self.field_proj = nn.Sequential(nn.Linear(field_dim, d), nn.GELU(), nn.Linear(d, 3 * d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.role_emb, self.side_emb, self.slot_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(6, d)
        self.mslot_emb, self.act_emb, self.fld_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(3, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, dropout=dropout, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        # 층별 어텐션 편향: 입력 4채널(배율, 데미지, KO, 공격기) -> 헤드별 값 / 숨은 기술 위협 2채널 -> 헤드별 값. 작게 시작
        self.bias_def = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_layers)])
        self.bias_off = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_layers)])
        self.bias_hid = nn.ModuleList([nn.Linear(2, n_heads) for _ in range(n_layers)])
        for lin in list(self.bias_def) + list(self.bias_off) + list(self.bias_hid):
            nn.init.normal_(lin.weight, std=0.5)
            nn.init.zeros_(lin.bias)
        self.turn_proj = nn.Linear(3 * d + field_dim, history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        hl = nn.TransformerEncoderLayer(history_dim, 4, history_dim * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.history_seq = nn.TransformerEncoder(hl, hist_layers, enable_nested_tensor=False)
        self.history_out = nn.Linear(history_dim, history_dim)
        self.fusion = nn.Sequential(nn.Linear(d + history_dim + G_DIM, latent_dim * 2), nn.LayerNorm(latent_dim * 2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.move_score = nn.Sequential(nn.Linear(latent_dim + 2 * d, head_width), nn.GELU(), nn.Linear(head_width, 2))
        self.switch_score = nn.Sequential(nn.Linear(latent_dim + d + CAND_N + G_DIM + 1, head_width), nn.GELU(), nn.Linear(head_width, 1))
        self.type_head = nn.Sequential(nn.Linear(latent_dim + CAND_N + 2 + G_DIM, 64), nn.GELU(), nn.Linear(64, 2))
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.value_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def _fit_field(self, fv):
        """field_vec 폭을 이 모델이 학습한 폭에 맞춤: 더 넓으면(인코더가 뒤에 열을 추가) 뒤를 버리고, 더 좁으면(옛 데이터) 0으로 채움"""
        n = self.field_proj[0].in_features
        return fv[..., :n] if fv.size(-1) >= n else F.pad(fv, (0, n - fv.size(-1)))

    def _history(self, turn_seq):
        B, T = turn_seq.size(0), turn_seq.size(1)
        pos = torch.arange(T, device=turn_seq.device).clamp(max=MAX_TURNS - 1)
        x = self.turn_proj(turn_seq) + self.turn_pos(pos)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=turn_seq.device), diagonal=1)
        return self.history_out(self.history_seq(x, mask=causal))

    @staticmethod
    def _turn_vec(enc):
        return torch.cat([enc["cls"], enc["active_mon"], enc["opp_active_mon"], enc["field"]], -1)

    def _bias(self, D, O, H, layer):
        """어텐션 편향 [B,H,64,64]: 내 포켓몬<->상대 기술 토큰(36..59), 상대 포켓몬<->내 기술 토큰(12..35), 내 포켓몬<->상대 포켓몬(숨은 기술 위협)"""
        B, NH = D.size(0), self.n_heads
        fd = self.bias_def[layer](D).permute(0, 3, 1, 2)                       # [B,H,6,24]
        fo = self.bias_off[layer](O).permute(0, 3, 1, 2)
        b = D.new_zeros(B, NH, N_TOK, N_TOK)
        b[:, :, 0:6, 36:60] = fd
        b[:, :, 36:60, 0:6] = fd.transpose(2, 3)
        b[:, :, 6:12, 12:36] = fo
        b[:, :, 12:36, 6:12] = fo.transpose(2, 3)
        fh = self.bias_hid[layer](H).permute(0, 3, 1, 2)                       # [B,heads,6(내),6(상대)]
        b[:, :, 0:6, 6:12] = fh
        b[:, :, 6:12, 0:6] = fh.transpose(2, 3)
        return b

    def _top_guess(self, logits, opp_cat, opp_mv, opp_num):
        """세트 추정의 기술 확률 -> 마리당 상위 K 후보. idx/w [B,6,K] (w = 세트 포함 확률 근사, 미공개 칸이 없으면 0), pseudo [B,6,K,46] (기울기 없음)"""
        with torch.no_grad():
            lg = logits["move"].detach().float()
            V = lg.size(-1)
            unk = self.belief.move.out_features - 2
            cur = opp_cat[..., 5:9].long()
            known = torch.zeros_like(lg, dtype=torch.bool)
            known.scatter_(2, torch.where((cur >= 1) & (cur < unk), cur, torch.zeros_like(cur)).clamp(0, V - 1), True)
            bad = torch.zeros(V, dtype=torch.bool, device=lg.device)
            bad[0], bad[unk:] = True, True
            prob = torch.softmax(lg.masked_fill(known | bad, -1e4), -1)
            present = opp_num[..., 11] > 0
            n_unrev = ((opp_mv.abs().sum(-1) == 0) & present[..., None]).sum(-1, keepdim=True).float()      # [B,6,1]
            topv, idx = prob.topk(K_GUESS, -1)
            w = (topv * n_unrev).clamp(max=1.0)                                      # 미공개 칸이 n개면 각 기술이 세트에 있을 확률 ~ n x 확률
            pseudo = self.move_table[idx] * (w > 0)[..., None]
        return {"idx": idx, "w": w, "pseudo": pseudo}

    @staticmethod
    def _legal_summary(leg, off_act, n_rev):
        """leg [B,22] (0/1), off_act [B,4,4] 활성 포켓몬의 기술 4칸 x (배율, 데미지, KO, 공격기) vs 상대 활성, n_rev [B] 공개된 활성 기술 수 -> [B, LEG_G]"""
        mv = leg[:, :4]
        n_normal, n_attack, n_sw = mv.sum(1), (mv * off_act[..., 3]).sum(1), leg[:, 8:14].sum(1)
        one_move = ((n_normal == 1) & (n_rev >= 2)).float()
        lock = (n_rev - n_normal).clamp_min(0) / 4.0                                     # 기술이 몇 개 막혀 있나 (고정·도발·PP 소진 등)
        full = off_act[..., :3].amax(1)
        legal_off = torch.where(mv[..., None] > 0, off_act[..., :3], torch.full_like(off_act[..., :3], -1.0)).amax(1).clamp_min(0)
        lost = torch.stack([(full[:, 1] - legal_off[:, 1]).clamp_min(0), (full[:, 2] - legal_off[:, 2]).clamp_min(0)], -1)
        return torch.cat([leg[:, :14], torch.stack([n_normal / 4.0, n_attack / 4.0, n_sw / 5.0, leg[:, 4:8].amax(1), one_move, lock], -1), legal_off, lost], -1)

    def _matchup_feats(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess, mult, stat, field_vec):
        B, K = my_cat.size(0), K_GUESS
        ar = torch.arange(B, device=my_cat.device)
        with torch.no_grad():
            ent = self.matchup.entry_frac(my_cat, my_num, field_vec)               # 장판 진입 피해 (v6~, 아니면 None)
            M = self.matchup(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, mult, entry=ent)
            D, O = M["def"], M["off"]                                              # 공개된 기술만 [B,6,24,4]
            Dp = self.matchup._pair(opp_cat, opp_num, guess["pseudo"], False, my_cat, my_num, True, att_mult=mult, entry=ent)   # [B,6(내),6K,4]
            w = guess["w"][:, None]                                                # [B,1,6(상대 o),K]
            dmg, ko = Dp[..., 1].view(B, 6, 6, K), Dp[..., 2].view(B, 6, 6, K)     # 후보별 데미지 비율/확정 KO [B,6(내 i),6(상대 o),K]
            H = torch.stack([1 - (1 - w * ko).prod(-1), (dmg * (w >= PLAUSIBLE)).amax(-1)], -1)   # [B,6,6,2] P_KO, 그럴듯한 최대 데미지
        act_m, act_o = my_num[:, :, 0].argmax(1), opp_num[:, :, 0].argmax(1)
        DS, OS = D.view(B, 6, 6, 4, 4), O.view(B, 6, 6, 4, 4)
        def_opp_act = DS[ar, :, act_o]                                             # [B,6(i),4,4] 상대 활성의 공개 기술이 내 포켓몬 i에게
        off_vs_opp_act = OS[ar, act_o]                                             # [B,6(i),4,4] 내 포켓몬 i의 기술이 상대 활성에게
        thr_on_my_act = DS[ar, act_m]                                              # [B,6(o),4,4] 상대 포켓몬 o의 공개 기술이 내 활성에게
        my_act_on_opp = OS[ar, :, act_m]                                           # [B,6(o),4,4] 내 활성 기술이 상대 포켓몬 o에게
        h_my, h_opp = H[ar, :, act_o], H[ar, act_m]                                # 상대 활성의 숨은 기술 -> 내 i / 상대 o의 숨은 기술 -> 내 활성 [B,6,6]
        mon_extra = torch.cat([
            torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], h_my, torch.zeros_like(stat)], -1),
            torch.cat([_mx(thr_on_my_act), _mx(my_act_on_opp), M["spd_opp"][..., None], h_opp, stat], -1)], 1)          # [B,12,15]
        mv_extra = torch.cat([off_vs_opp_act[..., :3], thr_on_my_act[..., :3]], 1)                                      # [B,12,4,3]
        bench_thr = DS[..., :3].amax((2, 3))[..., 1:]                                                                   # 후보 i가 상대 전체의 공개 기술에 맞는 최대 데미지/KO
        cand = torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], bench_thr, h_my], -1)         # [B,6,11]
        mon_extra = torch.cat([mon_extra, torch.cat([ent, torch.zeros_like(ent)], 1)[..., None]], -1)      # 진입 피해 비율 (상대 쪽 0)
        cand = torch.cat([cand, ent[..., None]], -1)
        return D, O, H, mon_extra, mv_extra, cand, ar, act_m

    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec, legal=None):
        field_vec = self._fit_field(field_vec)
        B, dev, d = my_team_cat.size(0), my_team_cat.device, self.d
        cat = torch.cat([my_team_cat[..., :10], opp_team_cat[..., :10]], 1).reshape(B * 12, 10)
        num = torch.cat([my_team_num, opp_team_num], 1).reshape(B * 12, -1)
        mvn = torch.cat([my_move_num, opp_move_num], 1).reshape(B * 12, 4, -1)
        E = self.embeddings
        parts = [E.species_proj(num[:, 11:18]), E._emb("item", cat[:, 0]), E._emb("ability", cat[:, 1]), E.type_embed(cat[:, 2]),
                 E.type_embed(cat[:, 3]), E.status_embed(cat[:, 4]), E.type_embed(cat[:, 9]), num[:, :11], num[:, 18:OLD_NUM]]
        raw = E._emb("move", cat[:, 5:9])
        revealed = (mvn.abs().sum(-1) > 0)
        n_rev = revealed.float().sum(-1, keepdim=True) / 4.0
        belief_logits = self.belief(opp_team_cat, opp_team_num)
        stat = belief_logits["stat"].detach().float()
        mult = torch.exp(STAT_SCALE * stat).clamp(*MULT_RANGE)
        guess = self._top_guess(belief_logits, opp_team_cat, opp_move_num, opp_team_num)
        D, O, H, mon_extra, mv_extra, cand, ar, act_m = self._matchup_feats(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num,
                                                                             opp_move_num, guess, mult, stat, field_vec)
        opp_act = opp_team_num[:, :, 0].argmax(1)
        # ---- 합법성: 토큰 플래그 + 요약 ----
        leg = torch.ones(B, 22, device=dev) if legal is None else legal.float()
        is_act = torch.zeros(B, 6, dtype=torch.bool, device=dev)
        is_act[ar, act_m] = True
        mon_flag = torch.ones(B, 12, device=dev)
        mon_flag[:, :6] = torch.where(is_act, torch.ones_like(leg[:, 8:14]), leg[:, 8:14])          # 활성은 해당 없음(1), 나머지는 교체 가능 여부
        mv_flag = torch.ones(B, 12, 4, device=dev)
        mv_flag[ar, act_m] = leg[:, :4]                                                              # 내 활성 포켓몬의 기술 4칸만 합법성이 의미 있음
        off_act = O.view(B, 6, 6, 4, 4)[ar, opp_act][ar, act_m]                                      # [B,4,4] 내 활성의 기술 4칸이 상대 활성에게
        n_rev_act = (my_move_num[ar, act_m].abs().sum(-1) > 0).sum(-1).float()
        leg_g = torch.cat([self._legal_summary(leg, off_act, n_rev_act), field_vec[:, EFFECT_START:EFFECT_START + EFFECT_DIM]], -1)      # 합법성 요약 + 양쪽 활성의 효과 열
        mon_in = torch.cat(parts + [mon_extra.reshape(B * 12, MON_X), n_rev, mon_flag.reshape(B * 12, 1)], -1)
        mv_in = torch.cat([E.move_encoder(raw, mvn), raw, mvn, mv_extra.reshape(B * 12, 4, 3), mv_flag.reshape(B * 12, 4, 1)], -1)
        side = torch.arange(12, device=dev) // 6
        slot = torch.arange(12, device=dev) % 6
        active = (num[:, 0] > 0.5).long().view(B, 12)
        base = self.side_emb(side)[None] + self.slot_emb(slot)[None] * self.pos_gate[None, :, None] + self.act_emb(active)
        g = torch.cat([guess["w"].sum(-1, keepdim=True), (guess["w"][..., None] * guess["pseudo"]).sum(2),
                       (guess["w"][..., None] * E._emb("move", guess["idx"]).detach()).sum(2)], -1)
        mon = self.mon_proj(mon_in).view(B, 12, d) + base + self.role_emb.weight[0]
        mon = mon + torch.cat([torch.zeros_like(mon[:, :6]), self.hid_proj(g)], 1)
        mv = self.move_proj(mv_in).view(B, 12, 4, d) + base[:, :, None] + self.mslot_emb.weight[None, None] * self.pos_gate[None, :, None, None] + self.role_emb.weight[1]
        fld = self.field_proj(field_vec).view(B, 3, d) + self.fld_emb.weight[None] + self.role_emb.weight[2]
        cls = self.cls.expand(B, 1, d) + self.role_emb.weight[3]
        x = torch.cat([mon, mv.reshape(B, 48, d), fld, cls], 1)
        mon_pad = (num[:, 11] == 0).view(B, 12)
        unrevealed = ~revealed.view(B, 12, 4) & ~mon_pad[:, :, None]
        x = torch.cat([x[:, :12], x[:, 12:60] + unrevealed.reshape(B, 48, 1) * self.unk_move, x[:, 60:]], 1)
        pad = torch.cat([mon_pad, mon_pad[:, :, None].expand(B, 12, 4).reshape(B, 48), mon_pad.new_zeros(B, 4)], 1)
        pad_f = torch.zeros(B, 1, 1, N_TOK, device=dev, dtype=x.dtype).masked_fill(pad[:, None, None, :], float("-inf"))
        for li, layer in enumerate(self.layers):
            mask = (self._bias(D, O, H, li).to(x.dtype) + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "belief_logits": belief_logits, "cand": cand,
                "field": field_vec, "act_m": act_m, "leg_g": leg_g, "leg_sw": leg[:, 8:14]}

    def _heads(self, enc, history, action_mask=None):
        B = history.size(0)
        lg = enc["leg_g"].to(history.dtype)
        latent = self.fusion(torch.cat([enc["cls"], history, lg], -1))
        cand = enc["cand"]
        ok = action_mask[:, 8:14] if action_mask is not None else torch.ones(B, 6, dtype=torch.bool, device=latent.device)
        act_feat = cand[torch.arange(B, device=latent.device), enc["act_m"]]
        best_off = torch.where(ok, cand[..., 4], torch.full_like(cand[..., 4], -1.0)).amax(1, keepdim=True).clamp_min(0)
        min_thr = torch.where(ok, cand[..., 7], torch.full_like(cand[..., 7], 9.0)).amin(1, keepdim=True)
        min_thr = torch.where(ok.any(1, keepdim=True), min_thr, torch.zeros_like(min_thr))
        mv_in = torch.cat([latent[:, None].expand(-1, 4, -1), enc["active_mv"], enc["active_mon"][:, None].expand(-1, 4, -1)], -1)
        sw_in = torch.cat([latent[:, None].expand(-1, 6, -1), enc["my_mon"], cand, lg[:, None].expand(-1, 6, -1), enc["leg_sw"][..., None].to(lg.dtype)], -1)
        mv, sw = self.move_score(mv_in), self.switch_score(sw_in).squeeze(-1)
        detail = torch.cat([mv[..., 0], mv[..., 1], sw, latent.new_full((B, 8), -1e9)], 1)
        type_logits = self.type_head(torch.cat([latent, act_feat, best_off, min_thr, lg], -1))
        if action_mask is not None:
            detail = detail.masked_fill(~action_mask, -1e9)
            no_move = ~(action_mask & ~self.is_switch).any(dim=1)
            no_switch = ~(action_mask & self.is_switch).any(dim=1)
            type_logits = type_logits.masked_fill(torch.stack([no_move, no_switch], dim=1), -1e9)
        lp_type = F.log_softmax(type_logits, -1)
        lp_move = F.log_softmax(detail.masked_fill(self.is_switch, -1e9), -1)
        lp_switch = F.log_softmax(detail.masked_fill(~self.is_switch, -1e9), -1)
        policy_logits = torch.where(self.is_switch, lp_type[:, 1:2] + lp_switch, lp_type[:, 0:1] + lp_move)
        bl = enc["belief_logits"]
        return {"policy_logits": policy_logits, "value": self.value_head(latent),
                "hid_move": bl["move"], "hid_item": bl["item"], "hid_ability": bl["ability"]}

    def forward(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                history_state=None, action_mask=None, legal_mask=None):
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                                legal=legal_mask if legal_mask is not None else action_mask)
        cur = self._turn_vec(enc).unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        out = self._heads(enc, self._history(seq)[:, -1], action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None, legal_mask=None):
        enc = self._encode_turn(*obs, legal=legal_mask if legal_mask is not None else action_mask)
        tv = self._turn_vec(enc)
        T = int(time_index.max().item()) + 1
        turn_seq = tv.new_zeros(n_seq, T, tv.size(-1))
        turn_seq[seq_index, time_index] = tv
        return self._heads(enc, self._history(turn_seq)[seq_index, time_index], action_mask)

    get_action = get_action
