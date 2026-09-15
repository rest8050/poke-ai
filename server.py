import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np

from model import DeepPokemonBattleTransformerNet

app = FastAPI(
    title="PokeAI Inference Server for Minecraft RCT Mod",
    description="배틀 세션별 GRU 히스토리 메모리가 유지되는 AI 추론 서버",
    version="3.0.0"
)

MODEL_PATH = "deep_checkpoint.pt"
model = DeepPokemonBattleTransformerNet()

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"✅ 딥 트랜스포머 모델 가중치 '{MODEL_PATH}' 로드 성공.")
except FileNotFoundError:
    print(f"⚠️ '{MODEL_PATH}' 가중치 파일이 없어 임의 초기화 상태로 작동합니다.")
    model.eval()

# 세션별 GRU 히스토리 상태 딕셔너리 (battle_id -> Tensor[1, 64])
history_sessions: Dict[str, torch.Tensor] = {}


def ensure_batch_tensor(data: Any, default_data: Any, dtype: torch.dtype, target_dims: int) -> torch.Tensor:
    """
    클라이언트에서 배치 차원([B])이 생략된 2D/1D 리스트가 올 때
    모델이 요구하는 차원([B, 6, 10], [B, 32] 등)으로 자동으로 unsqueeze(0) 보정
    """
    if data is None:
        t = torch.tensor(default_data, dtype=dtype)
    else:
        t = torch.tensor(data, dtype=dtype)
    
    while t.dim() < target_dims:
        t = t.unsqueeze(0)
    return t


class BattleStateRequest(BaseModel):
    battle_id: Optional[str] = "default_battle"
    trainer_name: Optional[str] = "Trainer_AI"
    # 카테고리 [6, 10] 또는 [1, 6, 10], 수치 [6, 12] 또는 [1, 6, 12], 기술 수치 [6, 4, 7] 또는 [1, 6, 4, 7]
    my_team_cat: Optional[Any] = None
    my_team_num: Optional[Any] = None
    my_move_num: Optional[Any] = None
    
    opp_team_cat: Optional[Any] = None
    opp_team_num: Optional[Any] = None
    opp_move_num: Optional[Any] = None
    
    field_vec: Optional[Any] = None
    action_mask: Optional[Any] = None


class DecisionResponse(BaseModel):
    action_type: str
    action_index: int
    chosen_action: str
    predicted_opponent_action: int
    confidence: float
    state_value: float


@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "service": "Deep Pokemon Battle AI Server v3.0", 
        "active_sessions": len(history_sessions)
    }


@app.post("/predict", response_model=DecisionResponse)
def predict_action(request: BattleStateRequest):
    """
    RCT Mod에서 호출하는 AI 행동 예측 엔드포인트 (배치 차원 자동 보정 & 세션별 GRU 턴 메모리 유지)
    """
    try:
        battle_id = request.battle_id or "default_battle"
        
        # 1. 텐서 배치 차원 보정 파싱 (Fix 3: List[List[int]] 차원 맞춤 오류 해결!)
        my_cat = ensure_batch_tensor(request.my_team_cat, [[0]*10]*6, torch.long, 3)
        my_num = ensure_batch_tensor(request.my_team_num, [[0.0]*12]*6, torch.float32, 3)
        my_m_num = ensure_batch_tensor(request.my_move_num, [[[[0.0]*7]*4]*6], torch.float32, 4)
        
        opp_cat = ensure_batch_tensor(request.opp_team_cat, [[0]*10]*6, torch.long, 3)
        opp_num = ensure_batch_tensor(request.opp_team_num, [[0.0]*12]*6, torch.float32, 3)
        opp_m_num = ensure_batch_tensor(request.opp_move_num, [[[[0.0]*7]*4]*6], torch.float32, 4)
        
        f_vec = ensure_batch_tensor(request.field_vec, [0.0]*48, torch.float32, 2)
        mask = ensure_batch_tensor(request.action_mask, [True]*22, torch.bool, 2)

        # 2. 세션별 이전 턴 GRU 히스토리 상태 가져오기
        prev_history = history_sessions.get(battle_id)

        # 3. 모델 추론
        action_idx, p_probs, opp_p_probs, val, new_history = model.get_action(
            my_cat, my_num, my_m_num,
            opp_cat, opp_num, opp_m_num,
            f_vec, history_state=prev_history, action_mask=mask
        )

        # 4. 새로운 턴 히스토리 세션에 저장
        history_sessions[battle_id] = new_history

        pred_opp_action = torch.argmax(opp_p_probs, dim=-1).item()
        confidence = float(p_probs[0, action_idx].item())

        # 5. 행동 해석
        if action_idx < 4:
            action_type = "move"
            target_idx = action_idx
            chosen_name = f"Use Move Slot {target_idx + 1}"
        elif 4 <= action_idx <= 7:
            action_type = "gimmick_move"
            target_idx = action_idx - 4
            chosen_name = f"Use Move Slot {target_idx + 1} with Terastallize/Dynamax"
        elif 8 <= action_idx <= 13:
            action_type = "switch"
            target_idx = action_idx - 8
            chosen_name = f"Switch to Team Slot {target_idx + 1}"
        else:
            action_type = "move"
            target_idx = 0
            chosen_name = "Default Action"

        return DecisionResponse(
            action_type=action_type,
            action_index=target_idx,
            chosen_action=chosen_name,
            predicted_opponent_action=pred_opp_action,
            confidence=confidence,
            state_value=float(val)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/reset_battle/{battle_id}")
def reset_battle_session(battle_id: str):
    """
    배틀이 종료되었을 때 세션의 GRU 히스토리를 초기화하는 엔드포인트
    """
    if battle_id in history_sessions:
        del history_sessions[battle_id]
        return {"status": "reset", "battle_id": battle_id}
    return {"status": "not_found", "battle_id": battle_id}


if __name__ == "__main__":
    print("🚀 Deep PokeAI v3.0 (세션 메모리 보존) 서버를 실행합니다... (http://localhost:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
