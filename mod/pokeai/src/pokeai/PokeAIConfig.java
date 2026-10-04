package pokeai;

/** 트레이너 JSON의 "ai": {"type": "pokeai", "data": {...}} 에서 읽는 설정 (Gson이 채움). */
public class PokeAIConfig {
    public String url = "http://127.0.0.1:8765";
    public int timeoutMs = 1000;        // 브리지 응답 대기 상한. 넘으면 이번 결정은 원래 RCT AI가 대신함
    public int breakerFails = 3;        // 연속 실패가 이 횟수를 넘으면
    public int breakerPauseSec = 30;    // 이 시간 동안 브리지를 건너뛰고 RCT AI만 씀 (서버 틱이 반복해서 멈추는 것을 방지)
    public boolean verbose = false;     // true면 매 결정을 pokeai-log/에 기록 (테스트용)
}
