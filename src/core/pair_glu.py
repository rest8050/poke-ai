"""v8 기술 x 포켓몬 쌍 모듈 (docs/team_attention_design.md 변형 C).
기술 토큰 48개(토큰 12..59) x 포켓몬 토큰 12개(토큰 0..11) = 576쌍마다 상호작용 벡터 z[k]를 만들어
(a) 어텐션 로짓 편향 [B,H,48,12] (기술<->포켓몬 양방향), (b) 기술 토큰으로의 집계 [B,48,d] 두 곳에 쓴다. 출력 투영은 0으로 시작.
mode: "glu" z = v(기술) * sigmoid(g(포켓몬) + 관계) | "bilin" z = v * (g + 관계) (시그모이드 없는 순수 곱) | "add" z = gelu(v + g + 관계) (파라미터 같은 덧셈형 대조군)
scope: "all" 576쌍 전부 | "syn" 같은 편의 소유자가 아닌 쌍만, 그 쌍에는 비공격기(변화기) 기술 값만 통과 (공격기끼리는 무관하다는 사전; 소유자 쌍은 가방이, 교차 편은 매치업 편향이 이미 담음)
관계 = 토큰 배치로 얻는 구조 값(0 소유자, 1 같은 편 다른 포켓몬, 2 상대 편): 슬롯 번호가 아니라 관계라서 my_pos=False 순열 대칭이 유지됨"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TOK = 64
MODES = ("glu", "bilin", "add")


class PairGLU(nn.Module):
    def __init__(self, d, k, n_heads, mode="glu", scope="all"):
        super().__init__()
        assert mode in MODES and scope in ("all", "syn"), (mode, scope)
        self.mode, self.scope = mode, scope
        self.ln = nn.LayerNorm(d)
        self.v, self.g, self.rel = nn.Linear(d, k), nn.Linear(d, k), nn.Embedding(3, k)
        self.to_bias, self.to_tok = nn.Linear(k, n_heads), nn.Linear(k, d)
        for lin in (self.to_bias, self.to_tok):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        owner = torch.arange(48) // 4                                   # 기술 토큰의 주인 포켓몬 (내 0..5, 상대 6..11)
        mon = torch.arange(12)
        rel = torch.where(owner[:, None] == mon[None], 0, torch.where(owner[:, None] // 6 == mon[None] // 6, 1, 2))
        self.register_buffer("rel_idx", rel, persistent=False)

    def forward(self, x, mon_pad, is_status=None):
        """x [B,64,d] 해당 층 입력, mon_pad [B,12] 빈 칸, is_status [B,48] 기술 토큰이 변화기(비공격기)면 1 (scope=syn에서만 씀).
        반환 (편향 [B,H,64,64] 전체 크기, 기술 토큰 더할 값 [B,48,d], 희소성 항 = 켜진 쌍의 평균 |z|)"""
        h = self.ln(x)
        v = self.v(h[:, 12:60])[:, :, None]                             # [B,48,1,k]
        g = self.g(h[:, :12])[:, None] + self.rel(self.rel_idx)[None]    # [B,48,12,k]
        if self.mode == "glu":
            z = v * torch.sigmoid(g)
        elif self.mode == "bilin":
            z = v * g
        else:
            z = F.gelu(v + g)
        if self.scope == "syn":
            on = (self.rel_idx == 1).to(z.dtype)[None] * is_status[:, :, None].to(z.dtype)      # [B,48,12]
            z = z * on[..., None]
            reg = z.abs().mean(-1).sum() / on.sum().clamp(min=1.0)
        else:
            reg = z.abs().mean()
        pb = self.to_bias(z).permute(0, 3, 1, 2)                         # [B,H,48,12]
        full = F.pad(pb, (0, N_TOK - 12, 12, N_TOK - 60))                # 행 12..59, 열 0..11
        keep = (~mon_pad).to(z.dtype)[:, None, :, None]
        return full + full.transpose(2, 3), self.to_tok((z * keep).sum(2) / 12.0), reg
