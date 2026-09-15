import asyncio
import os
import torch
import torch.optim as optim
from poke_env.player import RandomPlayer
from poke_env import LocalhostServerConfiguration
from player import SmartPokemonPlayer
from model import DeepPokemonBattleTransformerNet
from tensor_encoder import BattleTensorEncoder

class DataCollectingPlayer(SmartPokemonPlayer):
    def __init__(self, model, dataset_buffer, **kwargs):
        super().__init__(model=model, **kwargs)
        self.dataset_buffer = dataset_buffer

    def choose_move(self, battle):
        # A. 현재 턴 텐서 인코딩
        tensors = self.encoder.encode_battle(battle)
        my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask = tensors

        # B. 모델 추론을 통한 행동 선택
        order = super().choose_move(battle)

        # C. 선택한 행동의 Index 추정 (0~21)
        action_idx = 0
        if order.order and hasattr(order.order, 'id'):
            # 기술 사용 액션 (0~3)
            active_moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
            for idx, mv in enumerate(active_moves[:4]):
                if mv.id == order.order.id:
                    action_idx = idx
                    break
        elif order.order and hasattr(order.order, 'species'):
            # 교체 액션 (8~13)
            team_list = list(battle.team.values())
            for idx, pkmn in enumerate(team_list[:6]):
                if pkmn == order.order:
                    action_idx = 8 + idx
                    break

        # D. 수집 버퍼에 (State, Mask, Action_Label) 저장
        self.dataset_buffer.append({
            "my_team_cat": my_cat.squeeze(0).cpu(),
            "my_team_num": my_num.squeeze(0).cpu(),
            "my_move_num": my_m_num.squeeze(0).cpu(),
            "opp_team_cat": opp_cat.squeeze(0).cpu(),
            "opp_team_num": opp_num.squeeze(0).cpu(),
            "opp_move_num": opp_m_num.squeeze(0).cpu(),
            "field_vec": field_vec.squeeze(0).cpu(),
            "action_mask": action_mask.squeeze(0).cpu(),
            "action_label": torch.tensor(action_idx, dtype=torch.long)
        })

        return order


async def run_random_battle_training(n_battles=10):
    print(f"🚀 [gen9randombattle] 팀 생성 없이 {n_battles}판 자동 배틀 데이터 수집 시작...")

    model = DeepPokemonBattleTransformerNet()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    dataset_buffer = []

    ai_player = DataCollectingPlayer(
        model=model,
        dataset_buffer=dataset_buffer,
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=5
    )
    
    opponent_player = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=5
    )

    await ai_player.battle_against(opponent_player, n_battles=n_battles)

    print(f"✅ 배틀 완료! 총 {len(dataset_buffer)}개 턴(Turn) 데이터 수집 완료.")

    if len(dataset_buffer) == 0:
        print("수집된 데이터가 없습니다.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 수집된 데이터로 모델 학습 시작... (디바이스: {device})")
    model.to(device)
    model.train()
    
    batch_size = 8
    epochs = 50
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        
        # 셔플 추가 (간단하게 리스트 통째로 셔플)
        import random
        random.shuffle(dataset_buffer)
        
        for i in range(0, len(dataset_buffer), batch_size):
            batch = dataset_buffer[i:i+batch_size]
            if len(batch) < 4:
                continue

            my_cat = torch.stack([x["my_team_cat"] for x in batch]).to(device)
            my_num = torch.stack([x["my_team_num"] for x in batch]).to(device)
            my_m_num = torch.stack([x["my_move_num"] for x in batch]).to(device)
            opp_cat = torch.stack([x["opp_team_cat"] for x in batch]).to(device)
            opp_num = torch.stack([x["opp_team_num"] for x in batch]).to(device)
            opp_m_num = torch.stack([x["opp_move_num"] for x in batch]).to(device)
            field_vec = torch.stack([x["field_vec"] for x in batch]).to(device)
            action_mask = torch.stack([x["action_mask"] for x in batch]).to(device)
            target_actions = torch.stack([x["action_label"] for x in batch]).to(device)

            outputs = model(
                my_cat, my_num, my_m_num,
                opp_cat, opp_num, opp_m_num,
                field_vec, action_mask=action_mask
            )

            loss = torch.nn.functional.cross_entropy(outputs["policy_logits"], target_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        print(f"Epoch [{epoch}/{epochs}] | Avg Loss: {epoch_loss/batch_count:.4f}")

    torch.save(model.state_dict(), "deep_checkpoint_self_play.pt")
    print("🎉 학습 완료 및 'deep_checkpoint_self_play.pt' 가중치 저장 완수!")


if __name__ == "__main__":
    asyncio.run(run_random_battle_training(n_battles=50))
