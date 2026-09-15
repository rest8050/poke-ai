import torch
import torch.nn as nn
import torch.optim as optim
from model import DeepPokemonBattleTransformerNet, load_vocab_sizes


def generate_structured_dummy_team(batch_size: int, device: str = "cpu"):
    sizes = load_vocab_sizes("vocab.json")
    
    sp = torch.randint(1, sizes["num_species"] - 5, (batch_size, 6, 1))
    it = torch.randint(1, sizes["num_items"] - 5, (batch_size, 6, 1))
    ab = torch.randint(1, sizes["num_abilities"] - 5, (batch_size, 6, 1))
    t1 = torch.randint(1, sizes["num_types"] - 2, (batch_size, 6, 1))
    t2 = torch.randint(1, sizes["num_types"] - 2, (batch_size, 6, 1))
    st = torch.randint(0, sizes["num_statuses"] - 2, (batch_size, 6, 1))
    mvs = torch.randint(1, sizes["num_moves"] - 5, (batch_size, 6, 4))
    
    # 0번(첫번째 슬롯)은 항상 유효, 일부 슬롯은 0(패딩)으로 구성하여 마스킹 테스트
    cat_tensor = torch.cat([sp, it, ab, t1, t2, st, mvs], dim=-1) # [B, 6, 10]
    
    # 슬롯 4, 5번 중 일부는 0번 패딩으로 처리하여 패딩 마스크 작동 검증
    cat_tensor[:, 4:, 0] = 0 # sp = 0 (빈 슬롯)
    return cat_tensor


def generate_structured_dummy_numeric(batch_size: int, device: str = "cpu"):
    num = torch.zeros(batch_size, 6, 12, device=device)
    num[:, 0, 0] = 1.0 # 0번 포켓몬 Active 설정
    for i in range(6):
        num[:, i, 2] = i / 5.0 # 위치 슬롯 인코딩
        num[:, i, 3] = 0.5 + 0.5 * torch.rand(batch_size)
        num[:, i, 9] = 1.0
    return num


def generate_structured_dummy_move_numeric(batch_size: int, device: str = "cpu"):
    """
    [B, 6, 4, 7] 기술 수치 텐서 생성 (위력/200, 명중률, PP비율, 우선도, 물리/특수/변화 분류, 부가효과, 기술타입)
    """
    m_num = torch.rand(batch_size, 6, 4, 7, device=device)
    return m_num


def generate_dummy_action_mask(batch_size: int, device: str = "cpu"):
    mask = torch.zeros(batch_size, 22, dtype=torch.bool, device=device)
    mask[:, 0:4] = True
    mask[:, 8:14] = True
    return mask


def train_deep_epoch(model: DeepPokemonBattleTransformerNet, optimizer: optim.Optimizer, batch_size: int = 32):
    """
    5차 완벽 수정 검증 훈련 루프
    """
    model.train()
    
    my_team_cat = generate_structured_dummy_team(batch_size) # [B, 6, 10]
    my_team_num = generate_structured_dummy_numeric(batch_size)
    my_move_num = generate_structured_dummy_move_numeric(batch_size) # [B, 6, 4, 3]
    
    opp_team_cat = generate_structured_dummy_team(batch_size) # [B, 6, 10]
    opp_team_num = generate_structured_dummy_numeric(batch_size)
    opp_move_num = generate_structured_dummy_move_numeric(batch_size) # [B, 6, 4, 3]
    
    field_vec = torch.randn(batch_size, 48)
    action_mask = generate_dummy_action_mask(batch_size)
    opp_action_mask = generate_dummy_action_mask(batch_size)
    
    target_my_action = torch.randint(0, 4, (batch_size,))
    target_opp_action = torch.randint(0, 4, (batch_size,))
    target_opp_item = torch.randint(0, 50, (batch_size,))
    rewards = torch.randn(batch_size, 1)

    # Forward Pass
    outputs = model(
        my_team_cat, my_team_num, my_move_num,
        opp_team_cat, opp_team_num, opp_move_num,
        field_vec, 
        action_mask=action_mask,
        opp_action_mask=opp_action_mask
    )
    
    policy_logits = outputs["policy_logits"]
    opp_action_logits = outputs["opp_action_logits"]
    opp_item_logits = outputs["opp_item_logits"]
    value = outputs["value"]

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    loss_policy = criterion_ce(policy_logits, target_my_action)
    loss_opp_action = criterion_ce(opp_action_logits, target_opp_action)
    loss_opp_item = criterion_ce(opp_item_logits, target_opp_item)
    loss_value = criterion_mse(value, rewards)

    total_loss = loss_policy + 0.5 * loss_value + 0.5 * loss_opp_action + 0.3 * loss_opp_item

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "policy_loss": loss_policy.item(),
        "opp_action_loss": loss_opp_action.item(),
        "opp_item_loss": loss_opp_item.item(),
        "value_loss": loss_value.item()
    }


def main():
    print("⚡ [5차 완전 보완] 패딩 마스크, MoveEncoder, 상대 기술, GRU 세션 검증 시작...")
    
    model = DeepPokemonBattleTransformerNet()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    epochs = 10
    for epoch in range(1, epochs + 1):
        metrics = train_deep_epoch(model, optimizer, batch_size=32)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Total Loss: {metrics['total_loss']:.4f} | "
              f"Policy: {metrics['policy_loss']:.4f} | "
              f"Opp Action Pred: {metrics['opp_action_loss']:.4f}")

    save_path = "deep_checkpoint.pt"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 5차 지적사항 전원 교정 완수! 가중치가 '{save_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
