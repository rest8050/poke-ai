package pokeai;

import com.cobblemon.mod.common.api.battles.model.PokemonBattle;
import com.cobblemon.mod.common.api.battles.model.actor.BattleActor;
import com.cobblemon.mod.common.battles.ActiveBattlePokemon;
import com.cobblemon.mod.common.battles.BattleSide;
import com.cobblemon.mod.common.battles.InBattleMove;
import com.cobblemon.mod.common.battles.MoveActionResponse;
import com.cobblemon.mod.common.battles.MoveTarget;
import com.cobblemon.mod.common.battles.ShowdownActionResponse;
import com.cobblemon.mod.common.battles.ShowdownMoveset;
import com.cobblemon.mod.common.battles.SwitchActionResponse;
import com.cobblemon.mod.common.battles.Targetable;
import com.cobblemon.mod.common.battles.pokemon.BattlePokemon;
import com.gitlab.srcmc.rctapi.api.ai.RCTBattleAI;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import kotlin.jvm.functions.Function1;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/** 2~3단계: Cobblemon이 쌓은 Showdown 원본 메시지 덩어리를 파이썬 브리지에 보내 모델의 행동을 받아 응답으로 바꿈. 실패하면 첫 합법 행동으로 대체. */
@SuppressWarnings({"rawtypes", "unchecked"})
public class BridgeBattleAI extends HelloBattleAI {
    private final PokeAIConfig cfg;
    private final HttpClient http;
    private final RCTBattleAI rct = new RCTBattleAI();          // 브리지가 안 될 때 대신 행동하는 원래 RCT AI
    private final AtomicInteger fails = new AtomicInteger();
    private volatile long downUntil = 0;                         // 차단기: 이 시각(ms)까지 브리지를 건너뜀
    private final Gson gson = new Gson();
    private final java.util.Set<UUID> previewSent = ConcurrentHashMap.newKeySet();
    private final Map<UUID, Integer> sent = new ConcurrentHashMap<>();   // 배틀별로 이미 브리지에 보낸 덩어리 수 (전송 성공 때만 전진)

    public BridgeBattleAI(PokeAIConfig cfg) {
        super(cfg);
        this.cfg = cfg;
        this.http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).connectTimeout(Duration.ofMillis(Math.min(cfg.timeoutMs, 500))).build();
    }

    @Override
    public ShowdownActionResponse choose(ActiveBattlePokemon active, PokemonBattle battle, BattleSide aiSide,
                                         ShowdownMoveset moveset, boolean forceSwitch) {
        UUID id = battle.getBattleId();
        BattleActor actor = active.getActor();
        long t0 = System.nanoTime();
        if (System.currentTimeMillis() < downUntil) return rctChoose(active, battle, aiSide, moveset, forceSwitch, id);
        try {
            List<String> msgs = battle.getShowdownMessages();
            int from = sent.getOrDefault(id, 0);
            int to = msgs.size();
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("battle_id", id.toString());
            payload.put("side", actor.getShowdownId());
            payload.put("chunks", new ArrayList<>(msgs.subList(from, to)));
            if (previewSent.add(id)) {   // 배틀마다 한 번만: 상대(플레이어) 파티의 종/레벨 (Showdown 팀 프리뷰와 같은 수준의 정보)
                List<Map<String, Object>> preview = new ArrayList<>();
                for (BattleSide side : battle.getSides()) {
                    if (side == aiSide) continue;
                    for (BattleActor a : side.getActors()) {
                        for (BattlePokemon bp : a.getPokemonList()) {
                            Map<String, Object> m = new LinkedHashMap<>();
                            m.put("species", bp.getOriginalPokemon().showdownId());
                            m.put("level", bp.getOriginalPokemon().getLevel());
                            preview.add(m);
                        }
                    }
                }
                payload.put("opp_preview", preview);
            }
            HttpRequest req = HttpRequest.newBuilder(URI.create(cfg.url + "/choose"))
                    .timeout(Duration.ofMillis(cfg.timeoutMs))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(payload)))
                    .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) throw new RuntimeException("HTTP " + res.statusCode() + " " + res.body());
            fails.set(0);
            sent.put(id, to);   // 성공했을 때만 전진 → 실패하면 다음 호출에 같은 덩어리를 다시 보냄
            JsonObject r = gson.fromJson(res.body(), JsonObject.class);
            String kind = r.get("kind").getAsString();
            ShowdownActionResponse out = null;
            if (kind.equals("move") && !forceSwitch && moveset != null) {
                String moveId = r.get("id").getAsString();
                boolean wantTera = r.has("tera") && !r.get("tera").isJsonNull() && r.get("tera").getAsBoolean();
                String gimmickId = (wantTera && moveset.getCanTerastallize() != null) ? "terastal" : null;
                for (InBattleMove m : moveset.getMoves()) {
                    if (!m.getId().equals(moveId)) continue;
                    // 테라스탈은 기술 대상을 바꾸지 않음. gimmickMove는 Max/Z 기술용이라 그 대상을 쓰면 지진 등에 대상이 붙어 쇼다운이 거부(배틀 정지)
                    MoveTarget effTarget = m.getTarget();
                    String pnx = targetPnx(active, effTarget);
                    ShowdownActionResponse cand = new MoveActionResponse(m.getId(), pnx, gimmickId);
                    // Cobblemon의 isValid()는 self/all류에도 targetPnx가 유효 대상 목록에 있어야 한다고 오판함(자신 대상인데도
                    // 목록이 비어있지 않아 null이면 무조건 invalid) → 그런 타입은 canBeUsed()만으로 직접 판정
                    boolean ok = isClientChosenTarget(effTarget) ? cand.isValid(active, moveset, forceSwitch) : m.canBeUsed();
                    if (ok) out = cand;
                    break;
                }
            } else if (kind.equals("switch") && r.has("uuid") && !r.get("uuid").isJsonNull()) {
                UUID want = UUID.fromString(r.get("uuid").getAsString());
                for (BattlePokemon p : actor.getPokemonList()) {
                    if (p.getUuid().equals(want)) {
                        ShowdownActionResponse cand = new SwitchActionResponse(p.getUuid());
                        if (cand.isValid(active, moveset, forceSwitch)) out = cand;
                        break;
                    }
                }
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            if (cfg.verbose) log(id, String.format("turn=%d forceSwitch=%s bridge=%s -> %s (%.1fms)", battle.getTurn(), forceSwitch, res.body(),
                    out == null ? "적용 실패→RCT AI 대체" : out.getClass().getSimpleName(), ms));
            if (out != null) return out;
            return rctChoose(active, battle, aiSide, moveset, forceSwitch, id);
        } catch (Throwable t) {
            previewSent.remove(id);
            int n = fails.incrementAndGet();
            if (cfg.verbose) log(id, "bridge 실패→RCT AI 대체: " + t);
            if (n >= cfg.breakerFails) {
                downUntil = System.currentTimeMillis() + cfg.breakerPauseSec * 1000L;
                fails.set(0);
                PokeAIMod.LOG.warn("[pokeai] bridge failed {} times in a row ({}); using the standard RCT AI for {}s", n, t.toString(), cfg.breakerPauseSec);
            }
        }
        return rctChoose(active, battle, aiSide, moveset, forceSwitch, id);
    }

    /** 원래 RCT AI로 이번 결정을 대신함. 그것도 실패하면 첫 합법 행동 */
    private ShowdownActionResponse rctChoose(ActiveBattlePokemon active, PokemonBattle battle, BattleSide aiSide,
                                             ShowdownMoveset moveset, boolean forceSwitch, UUID id) {
        try {
            return rct.choose(active, battle, aiSide, moveset, forceSwitch);
        } catch (Throwable t) {
            PokeAIMod.LOG.warn("[pokeai] RCT AI fallback failed: {}", t.toString());
        }
        ShowdownActionResponse fb = firstValid(active, active.getActor(), moveset, forceSwitch, id);
        return fb != null ? fb : new com.cobblemon.mod.common.battles.DefaultActionResponse();
    }

    /** 클라이언트가 실제로 대상을 고를 수 있는 기술 타입만 pnx를 계산 (isClientChosenTarget은 HelloBattleAI에 정의).
     *  그 외(자신 대상, 전체 대상, 쇼다운이 자동으로 정하는 randomNormal/scripted 등)는 명시하면
     *  "Can't choose a target" 오류로 배틀이 멈추므로 null. */
    private String targetPnx(ActiveBattlePokemon active, MoveTarget t) {
        if (!isClientChosenTarget(t)) return null;
        try {
            Function1 f = t.getTargetList();
            Object r = f.invoke(active);
            if (r instanceof List && !((List) r).isEmpty()) return ((Targetable) ((List) r).get(0)).getPNX();
        } catch (Throwable ignored) {
        }
        return active.getOppositeOpponent().getPNX();
    }
}
