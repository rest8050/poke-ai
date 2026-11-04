"""정책/가치 네트워크 v8 (독립 구조, 상속 없음) = v7(= v6 + 특성 EXEC 규칙 + 합법 요약 슬롯 순서 통로 제거) + 숨은 특성/도구 기대 임베딩 + 융합층의 활성 토큰 직접 입력.
v7 대비 바뀐 점 (나머지는 v7과 같음):
A) 상대 쪽 숨겨진(공개 안 된) 특성/도구는 고정된 "숨김" 임베딩 대신 세트 추정 분포의 기대 임베딩 q @ 임베딩 표를 포켓몬 토큰에 넣음. 공개된 값/내 쪽/빈 칸은 원래 임베딩 그대로
   (표는 학습용 난수 novel_p/id_dropout 없이 embed.weight + proj(feat)로 결정론적으로 만들고, q는 detach라서 기울기는 임베딩 표로만 감)
D) 융합층 입력에 내 활성 / 상대 활성 포켓몬 토큰을 CLS와 함께 직접 넣음 (기술/교체 판단이 CLS 하나로만 전달받던 정보를 직접 봄): 융합 입력 d + history + G_DIM -> + 2d
H) 종류 헤드(기술/교체)의 은닉 폭을 옛 고정값 64에서 head_width(기본 192)로: 행동 점수 헤드와 같은 폭
v6 대비 바뀐 점 (v7, 나머지는 v6와 같음):
1) 매치업을 MatchupV7로: 정밀기술/두꺼운지방/옹골참, 부유류 면역을 상대 특성 확률(세트 추정) 기대값으로, 스피드 규칙(랭크·마비·스카프·순풍·날씨 특성·트릭룸)과 장난꾸러기 우선도로
   만든 선공 기대값 fm을 기술 토큰(양쪽)에, 위협 후보의 상대 능력 배율 entry_opp 3개를 교체 후보 특징에, 재생력 회복량 exit_heal을 전역 요약에 추가
2) 합법 요약에서 교체 6칸 비트를 뺌 (슬롯 순서대로 들어가 my_pos=False의 순열 대칭을 깨던 통로. 슬롯별 교체 가능 여부는 포켓몬 토큰의 플래그가 이미 줌): LEG_G 25 -> 19
상대 특성 분포는 세트 추정(ability) 로짓에서 빈칸/<unk>/숨김을 뺀 softmax, 공개된 것은 원핫. 상대 스카프 확률도 세트 추정(item)에서.
arch: {"model": "v8", "my_pos": false}
옛 구조는 model_v6.py(서버 탑재) (v6 퇴역 시 model_v6.py, matchup.py 삭제)
--- 이하 v6 설명 ---
현행 정책/가치 네트워크 v6 (독립 구조, 상속 없음): 엔티티 토큰 트랜스포머 + 규칙 기반 매치업 + 세트 추정 + 합법성/효과 입력.
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
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.belief import STAT_SCALE, SetBelief, build_move_table
from src.core.matchup_v7 import MatchupV7
from src.core.model import MAX_TURNS, DeepPokemonBattleTransformerNet, PokemonEmbeddingLayer
from src.core.pair_glu import PairGLU, PairMon
from src.core.tensor_encoder import EFFECT_DIM, EFFECT_START, FIELD_DIM, MOVE_NUM_DIM, NUM_DIM

# 3차원 어텐션 마스크(헤드별 편향)는 추론용 빠른 커널에서 학습 경로와 값이 달라짐(CLS 기준 2e-3) → 학습과 같은 일반 경로만 쓰도록 끔
torch.backends.mha.set_fastpath_enabled(False)

N_TOK = 64            # 포켓몬 12 + 기술 48 + 필드 3 + CLS 1
MV_EXTRA = 4            # 기술 토큰 추가 수치: 배율, 데미지, KO + 선공 기대값 fm
BASE_IN = 177 + NUM_DIM         # 포켓몬 토큰 기본 입력 차원 (종족값 투영 64 + 도구/특성/타입/상태/테라 임베딩 + 배틀 수치 + 이벤트 증거)
K_GUESS = 12          # 마리당 후보 기술 수 (숨은 기술 재현율: 상위 4 0.73 / 상위 8 0.94 / 상위 12 0.98)
PLAUSIBLE = 0.15      # 이 이상의 세트 포함 확률이면 "그럴듯한" 후보
MULT_RANGE = (0.6, 1.25)
LEG_G = 8 + 6 + 3 + 2    # 합법 비트 8(기술 4 + 테라 4) + 플래그 6 + 사용 가능한 기술만의 공격 요약 3 + 고정으로 잃은 화력 2
MON_X = 7 + 2 + 6 + 1    # 포켓몬 토큰 추가 수치: 매치업 요약 7 + 숨은 기술 위협 2 + 상대 능력치 예측 6 + 진입 피해 비율 1
CAND_N = 9 + 2 + 1 + 3   # 교체 후보 특징: 요약 9 + 상대 활성의 숨은 기술 위협 2 + 진입 피해 비율 1 + 위협 효과 3
PM_DIM, PM_DROPOUT = 16, 0.3   # prev_act: 마지막 기술 id 임베딩 폭, 드롭아웃
TERA_N, BENCH_N = 9, 3     # tera_feat: 기술칸마다 (테라 후 공격 3, 테라 전후 차 2, 상대 활성 위협 전/후 4) | bench_off: 교체 후보의 상대 벤치 공격 요약 3
G_DIM = LEG_G + EFFECT_DIM + 1   # 전역 요약: 합법성 요약 + 양쪽 활성의 효과 열 + 재생력 회복량


def _mx(x):           # [..., 4(기술), 4(채널)] 중 앞 3채널(배율, 데미지, KO)의 기술별 최댓값 → [..., 3]
    return x[..., :3].amax(-2)


class EntityPokemonNetV8(nn.Module):
    ACTION_DIM = 22
    uses_legal = True

    def __init__(self, vocab_path: str = "data/vocab.json", my_pos: bool = True, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0, history_dim: int = 256, hist_layers: int = 2, latent_dim: int = 256,
                 head_width: int = 192, entity_features: bool = True,
                 species_dim: int = 32, belief_hidden: int = 256, belief_species_dropout: float = 0.3, hist_aux: bool = False,
                 value_gammas: tuple = (), value_bins: int = 51, prev_act: bool = False, pm_fusion: bool = False, hist_only: bool = False,
                 loop_n: int = 0, loop_layers: int = 2, mon_moves=False, pair_glu: int = 0, pair_mode: str = "glu", pair_scope: str = "all",
                 tera_feat: bool = False, bench_off: bool = False, pair_feat: bool = False, pair_mon: int = 0):
        super().__init__()
        self.cfg = dict(model="v8", d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                        history_dim=history_dim, hist_layers=hist_layers, latent_dim=latent_dim, head_width=head_width,
                        species_dim=species_dim, belief_hidden=belief_hidden, belief_species_dropout=belief_species_dropout, my_pos=my_pos)
        if hist_aux:                  # 옛 체크포인트의 cfg와 같게 두려고 켰을 때만 기록
            self.cfg["hist_aux"] = True
        self.hist_aux = hist_aux
        self.mon_moves = mon_moves
        if mon_moves:                 # 포켓몬 토큰에 자기 기술 가방(공개된 기술 임베딩 합 + 숨은 칸은 세트 추정 기대값)을 직접 입력. 켰을 때만 cfg에 기록
            self.cfg["mon_moves"] = mon_moves     # True = 임베딩을 그대로 합, "mlp" = 기술마다 비선형 변환(48->96->48) 후 합
            self.bag_proj = nn.Linear(48, d_model)
            if mon_moves == "mlp":
                self.bag_mv = nn.Sequential(nn.Linear(48, 96), nn.GELU(), nn.Linear(96, 48))
        self.pair_k = pair_glu
        if pair_glu:                  # 기술 x 포켓몬 쌍 모듈(src/core/pair_glu.py, 층마다 하나): 켰을 때만 cfg에 기록
            self.cfg["pair_glu"], self.cfg["pair_mode"] = pair_glu, pair_mode
            if pair_scope != "all":
                self.cfg["pair_scope"] = pair_scope
            self.pair = nn.ModuleList([PairGLU(d_model, pair_glu, n_heads, pair_mode, pair_scope, 4 if pair_feat else 0) for _ in range(n_layers)])
        self.pair_mk = pair_mon
        if pair_mon:                  # 포켓몬 x 포켓몬 쌍 모듈(src/core/pair_glu.py PairMon, 층마다 하나): 켰을 때만 cfg에 기록
            self.cfg["pair_mon"] = pair_mon
            self.pair_m = nn.ModuleList([PairMon(d_model, pair_mon, n_heads) for _ in range(n_layers)])
        assert pair_glu or not pair_feat, "pair_feat는 pair_glu 필요"
        self.tera_feat, self.bench_off, self.pair_feat = tera_feat, bench_off, pair_feat
        for flag, on in (("tera_feat", tera_feat), ("bench_off", bench_off), ("pair_feat", pair_feat)):
            if on:                    # 켰을 때만 cfg에 기록
                self.cfg[flag] = True
        self.pair_reg = None          # 마지막 순전파의 쌍 희소성 항 (학습이 --pair-l1 배로 손실에 더함)
        self.prev_act = prev_act
        if prev_act:                  # 직전에 쓴 기술(내/상대) 입력: 켰을 때만 cfg에 기록
            self.cfg["prev_act"] = True
        assert prev_act or not pm_fusion, "pm_fusion은 prev_act 필요"
        self.pm_fusion, self.hist_only = pm_fusion, hist_only
        if pm_fusion:                 # 마지막 기술 임베딩을 융합층에도 직접 입력 (히스토리를 거치지 않는 경로)
            self.cfg["pm_fusion"] = True
        if hist_only:                 # 융합층이 현재 턴 CLS/활성 토큰을 직접 받지 않고 히스토리 출력(마지막 위치 = 현재 턴)만 받음: Metamon의 actor처럼 시퀀스 인코더 출력만 봄
            self.cfg["hist_only"] = True
        self.loop_n = loop_n
        if loop_n:                    # 반복 추론: 트렁크 뒤에 새 층 loop_layers개(가중치 공유)를 loop_n번 되풀이해 엔티티 토큰을 더 다듬음. 켰을 때만 cfg에 기록
            self.cfg["loop_n"], self.cfg["loop_layers"] = loop_n, loop_layers
        self.value_gammas = tuple(value_gammas)
        if self.value_gammas:         # 다중 γ 분류형(two-hot) 가치 헤드: γ마다 value_bins개 구간. 켰을 때만 cfg에 기록
            self.cfg["value_gammas"], self.cfg["value_bins"] = list(self.value_gammas), value_bins
        d = self.d = d_model
        field_dim = FIELD_DIM
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        pos = torch.ones(12) if my_pos else torch.tensor([0.0] * 6 + [1.0] * 6)       # 내 쪽 슬롯/기술 칸 순서는 임의(팀 export 순서)라서 끌 수 있음. 상대 쪽 순서는 공개된 순서라 정보가 있어 유지
        self.register_buffer("pos_gate", pos, persistent=False)
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        self.matchup = MatchupV7(vocab_path)
        self.belief = SetBelief(vocab_path, species_dim, belief_hidden, belief_species_dropout)
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
        if loop_n:      # 출력 투영을 0으로 시작: 처음엔 항등이라 v8 가중치에서 이어 학습해도 기존 정책이 그대로고, 반복이 쓸모 있을 때만 값이 생김
            self.loop = nn.ModuleList([nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, dropout=dropout, activation="gelu",
                                                                  batch_first=True, norm_first=True) for _ in range(loop_layers)])
            for layer in self.loop:
                for lin in (layer.self_attn.out_proj, layer.linear2):
                    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
            self.iter_emb = nn.Parameter(torch.zeros(loop_n, d))                 # 반복 번호 신호 (공유 층이 몇 번째 반복인지 알도록)
        self.norm = nn.LayerNorm(d)
        # 층별 어텐션 편향: 입력 4채널(배율, 데미지, KO, 공격기) -> 헤드별 값 / 숨은 기술 위협 2채널 -> 헤드별 값. 작게 시작
        n_bias = n_layers + (loop_layers if loop_n else 0)
        self.bias_def = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_bias)])
        self.bias_off = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_bias)])
        self.bias_hid = nn.ModuleList([nn.Linear(2, n_heads) for _ in range(n_bias)])
        for lin in list(self.bias_def) + list(self.bias_off) + list(self.bias_hid):
            nn.init.normal_(lin.weight, std=0.5)
            nn.init.zeros_(lin.bias)
        n_mv = len(json.load(open(vocab_path, encoding="utf-8"))["move"])
        self.n_mv = n_mv
        self.pm_emb, self.pm_drop = (nn.Embedding(2 * n_mv, PM_DIM), nn.Dropout(PM_DROPOUT)) if prev_act else (None, None)     # 마지막 기술 id를 저차원 임베딩+드롭아웃으로 규제 (내 id, 상대 id + n_mv). history_diag H 변형: 규제 없으면 과적합
        self.turn_proj = nn.Linear(3 * d + field_dim + (2 * PM_DIM if prev_act else 0), history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        hl = nn.TransformerEncoderLayer(history_dim, 4, history_dim * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.history_seq = nn.TransformerEncoder(hl, hist_layers, enable_nested_tensor=False)
        self.history_out = nn.Linear(history_dim, history_dim)
        if hist_aux:      # 기록 반영 숨김정보 보조 헤드: 세트 추정 로짓에 더하는 잔차 (0으로 시작해 처음엔 세트 추정과 같음). 정답(나중에 드러난 기술/도구/특성)이 상대의 지난 행동에서만 나올 수 있는 만큼 히스토리가 일을 하게 됨
            self.hist_hid = nn.ModuleDict({k: nn.Linear(history_dim, 6 * getattr(self.belief, k).out_features) for k in ("move", "item", "ability")})
            for m in self.hist_hid.values():
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        fuse_in = history_dim + G_DIM + (0 if hist_only else 3 * d) + (2 * PM_DIM if pm_fusion else 0)
        self.fusion =nn.Sequential(nn.Linear(fuse_in, latent_dim * 2), nn.LayerNorm(latent_dim * 2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.move_score = nn.Sequential(nn.Linear(latent_dim + 2 * d + (TERA_N if tera_feat else 0), head_width), nn.GELU(), nn.Linear(head_width, 2))
        self.switch_score = nn.Sequential(nn.Linear(latent_dim + d + CAND_N + (BENCH_N if bench_off else 0) + G_DIM + 1, head_width), nn.GELU(), nn.Linear(head_width, 1))
        self.type_head = nn.Sequential(nn.Linear(latent_dim + CAND_N + (BENCH_N if bench_off else 0) + 2 + G_DIM, head_width), nn.GELU(), nn.Linear(head_width, 2))
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.value_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, len(self.value_gammas) * value_bins if self.value_gammas else 1))
        if self.value_gammas:
            self.register_buffer("value_support", torch.linspace(-1, 1, value_bins), persistent=False)
            self.value_primary = min(range(len(self.value_gammas)), key=lambda i: abs(self.value_gammas[i] - 0.995))   # 기존 value(단일 γ=0.995)와 같은 의미로 내보낼 헤드

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

    def _tv(self, enc, prev_move):
        """히스토리 입력 턴 벡터. prev_act면 직전에 쓴 기술 id [B,2] (내, 상대; 0 = 없음/모름)의 임베딩을 이어 붙임"""
        tv = self._turn_vec(enc)
        if not self.prev_act:
            return tv
        if prev_move is None:
            prev_move = torch.zeros(tv.size(0), 2, dtype=torch.long, device=tv.device)
        e = self.pm_drop(self.pm_emb(prev_move + torch.tensor([0, self.n_mv], device=tv.device)).flatten(1)).to(tv.dtype)
        enc["pm"] = e
        return torch.cat([tv, e], -1)

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

    def _bench_off(self, O, opp_num, opp_act):
        """교체 후보 i의 공격이 상대 벤치(들어올 쪽, 활성 제외, 생존)에 통하는 정도 [B,6,3]: 최대 데미지, 평균 데미지, 평균 KO. 벤치가 없으면 0"""
        B = O.size(0)
        OS = O.view(B, 6, 6, 4, 4)                                              # [B,상대 o,내 i,기술,(배율,데미지,KO,공격기)]
        dmg, ko = OS[..., 1].amax(-1), OS[..., 2].amax(-1)                       # [B,o,i]
        ok = (opp_num[..., 11] > 0) & (opp_num[..., 2] > 0) & ~F.one_hot(opp_act, 6).bool()      # [B,6] 존재 + 생존 + 활성 아님
        w = ok.to(dmg.dtype)[:, :, None]
        n = w.sum(1).clamp_min(1.0)
        mx = (dmg * w).amax(1)
        return torch.stack([mx, (dmg * w).sum(1) / n, (ko * w).sum(1) / n], -1).float()

    def _tera_feat(self, D, O, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, field_vec, q_opp, p_scarf, mult, act_m, act_o, ar):
        """내 활성의 기술칸 4개마다 "지금 테라하면" 가정 매치업 [B,4,TERA_N]: 테라 후 상대 활성에 대한 (배율, 데미지, KO), 테라 전 대비 데미지/KO 차,
        상대 활성의 공개 기술이 내 활성에 주는 최대 (데미지, KO)의 테라 전/후. 매치업은 num[..,9](테라 중)일 때만 테라 타입을 쓰므로 내 쪽 테라 플래그를 켠 사본으로 한 번 더 계산"""
        B = my_cat.size(0)
        with torch.no_grad():
            my_t = my_num.clone()
            my_t[..., 9] = (my_cat[..., 9] > 0).float()
            Mt = self.matchup(my_cat, my_t, my_mv, opp_cat, opp_num, opp_mv, field_vec, q_opp, p_scarf, mult)
            off_n = O.view(B, 6, 6, 4, 4)[ar, act_o][ar, act_m].float()          # [B,4,4]
            off_t = Mt["off"].view(B, 6, 6, 4, 4)[ar, act_o][ar, act_m].float()
            thr_n = D.view(B, 6, 6, 4, 4)[ar, act_m][ar, act_o][..., 1:3].float().amax(1)      # [B,2] 최대 (데미지, KO)
            thr_t = Mt["def"].view(B, 6, 6, 4, 4)[ar, act_m][ar, act_o][..., 1:3].float().amax(1)
            return torch.cat([off_t[..., :3], (off_t - off_n)[..., 1:3], thr_n[:, None].expand(-1, 4, -1), thr_t[:, None].expand(-1, 4, -1)], -1)

    def _pair_feats(self, D, O, M, B):
        """쌍 모듈 수치 특징 [B,48,12,4]: 교차 편 쌍은 (배율, 데미지, KO) (내 기술 x 상대 포켓몬은 O, 상대 기술 x 내 포켓몬은 D), 나머지 0 / 4번째는 모든 쌍의 속도 대소 log2 비 (기술 주인 대 포켓몬)"""
        OS, DS = O.view(B, 6, 6, 4, 4), D.view(B, 6, 6, 4, 4)
        pf = O.new_zeros(B, 48, 12, 4)
        pf[:, :24, 6:, :3] = OS.permute(0, 2, 3, 1, 4).reshape(B, 24, 6, 4)[..., :3]      # [B,(내 i, 기술 m), 상대 o]
        pf[:, 24:, :6, :3] = DS.permute(0, 2, 3, 1, 4).reshape(B, 24, 6, 4)[..., :3]      # [B,(상대 o, 기술 m), 내 i]
        spe = torch.cat([M["spe_my"], M["spe_opp"]], 1).float()                            # [B,12]
        owner = torch.arange(48, device=O.device) // 4
        pf[..., 3] = (torch.log2(spe[:, owner, None].clamp_min(1.0) / spe[:, None, :].clamp_min(1.0)).clamp(-2, 2) / 2).to(pf.dtype)
        return pf

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
        """leg [B,22] (0/1), off_act [B,4,4] 활성 포켓몬의 기술 4칸 x (배율, 데미지, KO, 공격기) vs 상대 활성, n_rev [B] 공개된 활성 기술 수 -> [B, LEG_G]. 교체는 칸별 비트가 아니라 개수만 (슬롯 순서 통로 제거)"""
        mv = leg[:, :4]
        n_normal, n_attack, n_sw = mv.sum(1), (mv * off_act[..., 3]).sum(1), leg[:, 8:14].sum(1)
        one_move = ((n_normal == 1) & (n_rev >= 2)).float()
        lock = (n_rev - n_normal).clamp_min(0) / 4.0                                     # 기술이 몇 개 막혀 있나 (고정·도발·PP 소진 등)
        full = off_act[..., :3].amax(1)
        legal_off = torch.where(mv[..., None] > 0, off_act[..., :3], torch.full_like(off_act[..., :3], -1.0)).amax(1).clamp_min(0)
        lost = torch.stack([(full[:, 1] - legal_off[:, 1]).clamp_min(0), (full[:, 2] - legal_off[:, 2]).clamp_min(0)], -1)
        return torch.cat([leg[:, :8], torch.stack([n_normal / 4.0, n_attack / 4.0, n_sw / 5.0, leg[:, 4:8].amax(1), one_move, lock], -1), legal_off, lost], -1)

    @staticmethod
    def _prob(ids, logits):
        """상대 특성/도구 분포 [B,6,V]: 공개된 값은 원핫, 아니면 세트 추정 softmax (빈칸/<unk>/숨김 제외). ids [B,6], logits [B,6,V] (V = <unk>+2)"""
        lg = logits.detach().float().clone()
        unk = lg.size(-1) - 2
        lg[..., 0] = -1e9
        lg[..., unk:] = -1e9
        known = (ids >= 1) & (ids < unk)
        return torch.where(known[..., None], F.one_hot(ids.clamp(0, lg.size(-1) - 1), lg.size(-1)).float(), torch.softmax(lg, -1))

    def _expected_emb(self, kind, ids, q, B):
        """포켓몬 토큰 12개의 특성/도구 임베딩 [B*12, D]. 상대 쪽(뒤 6마리)에서 숨겨진 값(<unk>+1)만 분포 q [B,6,V]의 기대 임베딩 q @ 표로 바꾸고, 나머지는 원래 임베딩.
        표는 학습용 난수(novel_p, id_dropout)를 뺀 embed.weight + proj(feat)"""
        E = self.embeddings
        e = E._emb(kind, ids).view(B, 12, -1)
        w = getattr(E, f"{kind}_embed").weight
        if E.has_feat:
            w = w + getattr(E, f"{kind}_proj")(getattr(E, f"{kind}_feat"))
        hidden = (ids.view(B, 12)[:, 6:] == E.unk[kind] + 1)[..., None]
        expected = (q @ w[: q.size(-1)].to(q.dtype)).to(e.dtype)
        return torch.cat([e[:, :6], torch.where(hidden, expected, e[:, 6:])], 1).view(B * 12, -1)

    def _matchup_feats(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess, mult, stat, field_vec, q_opp, p_scarf):
        B, K = my_cat.size(0), K_GUESS
        ar = torch.arange(B, device=my_cat.device)
        with torch.no_grad():
            M = self.matchup(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, field_vec, q_opp, p_scarf, mult)
            D, O, ent, entry_opp = M["def"], M["off"], M["entry_frac"], M["entry_opp"]    # 공개된 기술만 [B,6,24,4]
            Dp = self.matchup._pair(opp_cat, opp_num, guess["pseudo"], False, q_opp, my_cat, my_num, True, self.matchup._onehot(my_cat),
                                    att_mult=mult, entry=ent, cand_f=(entry_opp[..., 0], entry_opp[..., 1]), mid=guess["idx"], field=field_vec)     # [B,6(내),6K,4]
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
        mv_extra = torch.cat([mv_extra, torch.cat([M["fm_my"], M["fm_opp"]], 1)[..., None]], -1)                       # [B,12,4,4] + 선공 기대값
        bench_thr = DS[..., :3].amax((2, 3))[..., 1:]                                                                   # 후보 i가 상대 전체의 공개 기술에 맞는 최대 데미지/KO
        cand = torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], bench_thr, h_my], -1)         # [B,6,11]
        mon_extra = torch.cat([mon_extra, torch.cat([ent, torch.zeros_like(ent)], 1)[..., None]], -1)      # 진입 피해 비율 (상대 쪽 0)
        cand = torch.cat([cand, ent[..., None], entry_opp], -1)                                                         # [B,6,15]
        return D, O, H, mon_extra, mv_extra, cand, ar, act_m, M["exit_heal"], M

    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec, legal=None):
        field_vec = self._fit_field(field_vec)
        B, dev, d = my_team_cat.size(0), my_team_cat.device, self.d
        cat = torch.cat([my_team_cat[..., :10], opp_team_cat[..., :10]], 1).reshape(B * 12, 10)
        num = torch.cat([my_team_num, opp_team_num], 1).reshape(B * 12, -1)
        mvn = torch.cat([my_move_num, opp_move_num], 1).reshape(B * 12, 4, -1)
        E = self.embeddings
        belief_logits = self.belief(opp_team_cat, opp_team_num)
        q_opp, q_item = self._prob(opp_team_cat[..., 1], belief_logits["ability"]), self._prob(opp_team_cat[..., 0], belief_logits["item"])
        p_scarf = q_item[..., self.matchup.scarf]
        parts = [E.species_proj(num[:, 11:18]), self._expected_emb("item", cat[:, 0], q_item, B), self._expected_emb("ability", cat[:, 1], q_opp, B),
                 E.type_embed(cat[:, 2]), E.type_embed(cat[:, 3]), E.status_embed(cat[:, 4]), E.type_embed(cat[:, 9]), num[:, :11], num[:, 18:]]
        raw = E._emb("move", cat[:, 5:9])
        revealed = (mvn.abs().sum(-1) > 0)
        n_rev = revealed.float().sum(-1, keepdim=True) / 4.0
        stat = belief_logits["stat"].detach().float()
        mult = torch.exp(STAT_SCALE * stat).clamp(*MULT_RANGE)
        guess = self._top_guess(belief_logits, opp_team_cat, opp_move_num, opp_team_num)
        D, O, H, mon_extra, mv_extra, cand, ar, act_m, exit_heal, Mx = self._matchup_feats(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num,
                                                                             opp_move_num, guess, mult, stat, field_vec, q_opp, p_scarf)
        opp_act = opp_team_num[:, :, 0].argmax(1)
        if self.bench_off:
            cand = torch.cat([cand, self._bench_off(O, opp_team_num, opp_act)], -1)
        tera_f = self._tera_feat(D, O, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec, q_opp, p_scarf, mult, act_m, opp_act, ar) if self.tera_feat else None
        pf = self._pair_feats(D, O, Mx, B) if self.pair_feat else None
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
        leg_g = torch.cat([self._legal_summary(leg, off_act, n_rev_act), field_vec[:, EFFECT_START:EFFECT_START + EFFECT_DIM], exit_heal[:, None]], -1)      # 합법성 요약 + 양쪽 활성의 효과 열 + 재생력 회복량
        mon_in = torch.cat(parts + [mon_extra.reshape(B * 12, MON_X), n_rev, mon_flag.reshape(B * 12, 1)], -1)
        mv_in = torch.cat([E.move_encoder(raw, mvn), raw, mvn, mv_extra.reshape(B * 12, 4, MV_EXTRA), mv_flag.reshape(B * 12, 4, 1)], -1)
        side = torch.arange(12, device=dev) // 6
        slot = torch.arange(12, device=dev) % 6
        active = (num[:, 0] > 0.5).long().view(B, 12)
        base = self.side_emb(side)[None] + self.slot_emb(slot)[None] * self.pos_gate[None, :, None] + self.act_emb(active)
        g = torch.cat([guess["w"].sum(-1, keepdim=True), (guess["w"][..., None] * guess["pseudo"]).sum(2),
                       (guess["w"][..., None] * E._emb("move", guess["idx"]).detach()).sum(2)], -1)
        mon = self.mon_proj(mon_in).view(B, 12, d) + base + self.role_emb.weight[0]
        mon = mon + torch.cat([torch.zeros_like(mon[:, :6]), self.hid_proj(g)], 1)
        if self.mon_moves:      # 기술 가방: 양쪽 같은 경로. 내 쪽은 전부 공개라 숨은 항이 0, 상대는 공개되면 실제 임베딩이 들어가고 숨은 칸 기대값은 그만큼 줄어듦
            bag = ((self.bag_mv(raw) if self.mon_moves == "mlp" else raw) * revealed[..., None]).sum(1).view(B, 12, -1)
            bag = bag + torch.cat([torch.zeros_like(bag[:, :6]), g[..., -48:].to(bag.dtype)], 1)
            mon = mon + self.bag_proj(bag)
        mv = self.move_proj(mv_in).view(B, 12, 4, d) + base[:, :, None] + self.mslot_emb.weight[None, None] * self.pos_gate[None, :, None, None] + self.role_emb.weight[1]
        fld = self.field_proj(field_vec).view(B, 3, d) + self.fld_emb.weight[None] + self.role_emb.weight[2]
        cls = self.cls.expand(B, 1, d) + self.role_emb.weight[3]
        x = torch.cat([mon, mv.reshape(B, 48, d), fld, cls], 1)
        mon_pad = (num[:, 11] == 0).view(B, 12)
        unrevealed = ~revealed.view(B, 12, 4) & ~mon_pad[:, :, None]
        x = torch.cat([x[:, :12], x[:, 12:60] + unrevealed.reshape(B, 48, 1) * self.unk_move, x[:, 60:]], 1)
        pad = torch.cat([mon_pad, mon_pad[:, :, None].expand(B, 12, 4).reshape(B, 48), mon_pad.new_zeros(B, 4)], 1)
        pad_f = torch.zeros(B, 1, 1, N_TOK, device=dev, dtype=x.dtype).masked_fill(pad[:, None, None, :], float("-inf"))
        is_status = (mvn[..., 4] > 0.9).view(B, 48).float() if self.pair_k else None      # 공개된 변화기 기술 (기술 수치 5번째 = 분류, 변화기 1.0)
        regs = []
        for li, layer in enumerate(self.layers):
            bias = self._bias(D, O, H, li).to(x.dtype)
            if self.pair_k:
                pb, pt, rg = self.pair[li](x, mon_pad, is_status, pf)
                regs.append(rg)
                bias = bias + pb.to(x.dtype)
                x = torch.cat([x[:, :12], x[:, 12:60] + pt.to(x.dtype), x[:, 60:]], 1)
            if self.pair_mk:
                pbm, ptm = self.pair_m[li](x, mon_pad)
                bias = bias + pbm.to(x.dtype)
                x = torch.cat([x[:, :12] + ptm.to(x.dtype), x[:, 12:]], 1)
            mask = (bias + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        if regs:
            self.pair_reg = torch.stack(regs).mean().float()
        if self.loop_n:
            masks = [(self._bias(D, O, H, len(self.layers) + j).to(x.dtype) + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK) for j in range(len(self.loop))]   # 반복마다 같으니 한 번만
            for t in range(self.loop_n):
                x = x + self.iter_emb[t]
                for layer, mask in zip(self.loop, masks):
                    x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "belief_logits": belief_logits, "cand": cand,
                "field": field_vec, "act_m": act_m, "leg_g": leg_g, "leg_sw": leg[:, 8:14], "tera_feat": tera_f}

    def _heads(self, enc, history, action_mask=None):
        B = history.size(0)
        lg = enc["leg_g"].to(history.dtype)
        cur = [] if self.hist_only else [enc["cls"]]
        tail = [] if self.hist_only else [enc["active_mon"], enc["opp_active_mon"]]
        pm = [enc["pm"].to(history.dtype)] if self.pm_fusion else []
        latent = self.fusion(torch.cat(cur + [history, lg] + tail + pm, -1))
        cand = enc["cand"]
        ok = action_mask[:, 8:14] if action_mask is not None else torch.ones(B, 6, dtype=torch.bool, device=latent.device)
        act_feat = cand[torch.arange(B, device=latent.device), enc["act_m"]]
        best_off = torch.where(ok, cand[..., 4], torch.full_like(cand[..., 4], -1.0)).amax(1, keepdim=True).clamp_min(0)
        min_thr = torch.where(ok, cand[..., 7], torch.full_like(cand[..., 7], 9.0)).amin(1, keepdim=True)
        min_thr = torch.where(ok.any(1, keepdim=True), min_thr, torch.zeros_like(min_thr))
        mv_in = torch.cat([latent[:, None].expand(-1, 4, -1), enc["active_mv"], enc["active_mon"][:, None].expand(-1, 4, -1)], -1)
        if self.tera_feat:
            mv_in = torch.cat([mv_in, enc["tera_feat"].to(mv_in.dtype)], -1)
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
        out = {"policy_logits": policy_logits, "latent": latent, "value": self.value_head(latent),
               "hid_move": bl["move"], "hid_item": bl["item"], "hid_ability": bl["ability"]}
        if self.value_gammas:
            out["value_logits"] = out["value"].view(B, len(self.value_gammas), -1)
            out["value"] = (out["value_logits"].float().softmax(-1) * self.value_support).sum(-1)[:, self.value_primary:self.value_primary + 1]
        return out

    def forward(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                history_state=None, action_mask=None, legal_mask=None, prev_move=None):
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                                legal=legal_mask if legal_mask is not None else action_mask)
        cur = self._tv(enc, prev_move).unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        out = self._heads(enc, self._history(seq)[:, -1], action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None, legal_mask=None, prev_move=None):
        enc = self._encode_turn(*obs, legal=legal_mask if legal_mask is not None else action_mask)
        tv = self._tv(enc, prev_move)
        T = int(time_index.max().item()) + 1
        turn_seq = tv.new_zeros(n_seq, T, tv.size(-1))
        turn_seq[seq_index, time_index] = tv
        out = self._heads(enc, self._history(turn_seq)[seq_index, time_index], action_mask)
        if self.hist_aux:   # 보조 가지는 턴 벡터를 detach해서 트렁크를 건드리지 않고, 히스토리 모듈과 이 헤드만 학습 (훈련 전용: 추론 경로에는 안 나옴)
            h = self._history(turn_seq.detach())[seq_index, time_index]
            for k in ("move", "item", "ability"):
                out[f"hidh_{k}"] = enc["belief_logits"][k].detach() + self.hist_hid[k](h).view(h.size(0), 6, -1)
        return out

    get_action = DeepPokemonBattleTransformerNet.get_action
