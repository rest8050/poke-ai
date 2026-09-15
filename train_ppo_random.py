import asyncio
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from poke_env.player import RandomPlayer
from poke_env import LocalhostServerConfiguration
from player import SmartPokemonPlayer
from model import DeepPokemonBattleTransformerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOCollectingPlayer(SmartPokemonPlayer):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.trajectories = {}
        self.last_state_scores = {}
        self.is_eval = False

    def clear_memory(self):
        self.trajectories.clear()
        self.last_state_scores.clear()
        self.history.clear()
        self.battles.clear()

    def compute_dense_reward(self, battle):
        if not hasattr(battle, "team") or not battle.team:
            return 0.0
            
        my_fainted = sum(1 for mon in battle.team.values() if mon.fainted)
        opp_fainted = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
        
        my_hp = sum(mon.current_hp_fraction for mon in battle.team.values())
        revealed_opp = len(battle.opponent_team)
        unrevealed_opp = 6 - revealed_opp
        opp_hp = sum(mon.current_hp_fraction for mon in battle.opponent_team.values()) + unrevealed_opp
        
        return ((my_hp - opp_hp) + (opp_fainted * 2.0 - my_fainted * 2.0)) * 0.1

    def choose_move(self, battle):
        tag = battle.battle_tag
        current_score = self.compute_dense_reward(battle)
        
        if not self.is_eval:
            if tag not in self.trajectories:
                self.trajectories[tag] = []
            else:
                last_score = self.last_state_scores.get(tag, 0.0)
                step_reward = current_score - last_score
                if len(self.trajectories[tag]) > 0:
                    self.trajectories[tag][-1]["reward"] = step_reward
                    
            self.last_state_scores[tag] = current_score

        tensors = self.encoder.encode_battle(battle)
        my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask = [t.to(device) for t in tensors]

        history = self.history.get(tag)
        if history is not None:
            history = history.to(device)
            
        action_idx, log_prob, val, new_history = self.model.get_action_rl(
            my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, history, action_mask, deterministic=self.is_eval
        )
        self.history[tag] = new_history

        if not self.is_eval:
            self.trajectories[tag].append({
                "my_team_cat": my_cat.squeeze(0),
                "my_team_num": my_num.squeeze(0),
                "my_move_num": my_m_num.squeeze(0),
                "opp_team_cat": opp_cat.squeeze(0),
                "opp_team_num": opp_num.squeeze(0),
                "opp_move_num": opp_m_num.squeeze(0),
                "field_vec": field_vec.squeeze(0),
                "action_mask": action_mask.squeeze(0),
                "history_in": history.squeeze(0).detach() if history is not None else torch.zeros(64, device=device),
                "action": action_idx,
                "log_prob": log_prob.squeeze(0),
                "value": val.squeeze(0)
            })

        active_pkmn = battle.active_pokemon
        my_team_list = list(battle.team.values()) if hasattr(battle, "team") else []

        if action_idx < 8 and active_pkmn:
            move_idx = action_idx % 4
            active_moves = list(active_pkmn.moves.values())
            avail_ids = {m.id for m in battle.available_moves}
            if move_idx < len(active_moves) and active_moves[move_idx].id in avail_ids:
                target_move = active_moves[move_idx]
                z_ids = {m.id for m in active_pkmn.available_z_moves} if battle.can_z_move else set()
                z_move = action_idx >= 4 and target_move.id in z_ids
                mega = not z_move and battle.can_mega_evolve
                dynamax = not z_move and not mega and battle.can_dynamax
                return self.create_order(target_move, z_move=z_move, mega=mega, dynamax=dynamax)

        elif 8 <= action_idx <= 13:
            slot_idx = action_idx - 8
            if slot_idx < len(my_team_list):
                target_pkmn = my_team_list[slot_idx]
                if target_pkmn in battle.available_switches:
                    return self.create_order(target_pkmn)

        return self.choose_random_move(battle)

def compute_gae(trajectory, final_reward, gamma=0.99, lam=0.95):
    values = [step["value"] for step in trajectory]
    values.append(torch.tensor([0.0], device=device))
    
    advantages = []
    gae = 0
    for t in reversed(range(len(trajectory))):
        reward = trajectory[t].get("reward", 0.0)
        if t == len(trajectory) - 1:
            reward += final_reward 
            
        delta = reward + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        
    return advantages

async def run_ppo_training(n_iterations=1000, battles_per_iter=100, batch_size=256, ppo_epochs=4):
    print(f"🚀 [PPO] Continuous Loop 시작 (총 {n_iterations} Iterations, Device: {device})")
    
    model = DeepPokemonBattleTransformerNet()
    if os.path.exists("deep_checkpoint_ppo.pt"):
        model.load_state_dict(torch.load("deep_checkpoint_ppo.pt", map_location=device))
        print("기존 PPO 가중치 로드 완료! (이어서 학습)")
    model.to(device)
    
    # 🌟 플레이어 객체는 루프 바깥에서 단 한 번만 생성하여 웹소켓/세션 충돌(유령 챌린지) 방지
    ai_player = PPOCollectingPlayer(
        model=model,
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )
    opp_player = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )
    
    # 옵티마이저를 루프 바깥으로 빼서 Adam의 모멘텀(기억)이 유지되도록 수정
    optimizer = optim.Adam(model.parameters(), lr=2e-4) 
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n==================================================")
        print(f"🔥 Iteration [{iteration}/{n_iterations}] - {battles_per_iter}판 데이터 수집 시작")
        print(f"==================================================")
        
        # 이전 이터레이션의 찌꺼기 딕셔너리 비우기
        ai_player.clear_memory()
        opp_player.battles.clear()
        ai_player.is_eval = False
        
        await ai_player.battle_against(opp_player, n_battles=battles_per_iter)
        
        dataset_buffer = []
        win_count = 0
        
        for tag, trajectory in ai_player.trajectories.items():
            if len(trajectory) == 0: continue
            
            battle_obj = ai_player.battles.get(tag)
            if not battle_obj or battle_obj.won is None:
                continue
                
            reward = 1.0 if battle_obj.won else -1.0
            if battle_obj.won: win_count += 1
            
            advantages = compute_gae(trajectory, reward)
            
            for t, step in enumerate(trajectory):
                step["advantage"] = advantages[t].detach()
                step["return"] = (advantages[t] + step["value"]).detach()
                dataset_buffer.append(step)
                
        print(f"✅ 수집 완료! {win_count}승 / {battles_per_iter}판 | 수집된 총 턴 수: {len(dataset_buffer)}")
        
        if len(dataset_buffer) == 0:
            continue
            
        print("🧠 PPO 가중치 업데이트 시작...")
        
        clip_epsilon = 0.2
        value_coef = 0.25
        entropy_coef = 0.002
        
        import random
        
        for epoch in range(ppo_epochs):
            model.eval()
            random.shuffle(dataset_buffer)
            
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            batch_count = 0
            
            for i in range(0, len(dataset_buffer), batch_size):
                batch = dataset_buffer[i:i+batch_size]
                if len(batch) < 4: continue
                
                my_cat = torch.stack([x["my_team_cat"] for x in batch])
                my_num = torch.stack([x["my_team_num"] for x in batch])
                my_m_num = torch.stack([x["my_move_num"] for x in batch])
                opp_cat = torch.stack([x["opp_team_cat"] for x in batch])
                opp_num = torch.stack([x["opp_team_num"] for x in batch])
                opp_m_num = torch.stack([x["opp_move_num"] for x in batch])
                field_vec = torch.stack([x["field_vec"] for x in batch])
                action_mask = torch.stack([x["action_mask"] for x in batch])
                history_in = torch.stack([x["history_in"] for x in batch])
                
                old_actions = torch.tensor([x["action"] for x in batch], device=device)
                old_log_probs = torch.stack([x["log_prob"] for x in batch])
                advantages = torch.stack([x["advantage"] for x in batch]).view(-1)
                returns = torch.stack([x["return"] for x in batch]).view(-1, 1)
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                outputs = model(
                    my_cat, my_num, my_m_num,
                    opp_cat, opp_num, opp_m_num,
                    field_vec, history_state=history_in, action_mask=action_mask
                )
                
                p_logits = outputs["policy_logits"]
                values = outputs["value"]
                
                dist = torch.distributions.Categorical(logits=p_logits)
                new_log_probs = dist.log_prob(old_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, returns)
                
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                batch_count += 1
                
            print(f"🔄 Epoch [{epoch+1}/{ppo_epochs}] P-Loss: {epoch_policy_loss/batch_count:.4f} | V-Loss: {epoch_value_loss/batch_count:.4f} | Ent: {epoch_entropy/batch_count:.4f}")
    
        torch.save(model.state_dict(), "deep_checkpoint_ppo.pt")
        print("💾 가중치 저장 완료!")

        if iteration % 10 == 0:
            print(f"\n==================================================")
            print(f"🏆 [평가 모드] 모델 실력 검증 (50판 승부)")
            print(f"==================================================")
            
            ai_player.is_eval = True
            ai_player.clear_memory()
            opp_player.battles.clear()
            
            await ai_player.battle_against(opp_player, n_battles=50)
            
            eval_wins = sum(1 for b in ai_player.battles.values() if b.won)
            print(f"📊 평가 결과 승률: {eval_wins}승 / 50판 ({eval_wins/50*100:.1f}%)")
            print(f"==================================================")

if __name__ == "__main__":
    asyncio.run(run_ppo_training(n_iterations=1000, battles_per_iter=100, batch_size=256, ppo_epochs=4))
