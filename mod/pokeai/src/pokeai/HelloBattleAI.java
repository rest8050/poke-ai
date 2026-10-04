package pokeai;

import com.cobblemon.mod.common.api.battles.model.PokemonBattle;
import com.cobblemon.mod.common.api.battles.model.actor.BattleActor;
import com.cobblemon.mod.common.api.battles.model.ai.BattleAI;
import com.cobblemon.mod.common.battles.ActiveBattlePokemon;
import com.cobblemon.mod.common.battles.BattleSide;
import com.cobblemon.mod.common.battles.DefaultActionResponse;
import com.cobblemon.mod.common.battles.InBattleMove;
import com.cobblemon.mod.common.battles.MoveActionResponse;
import com.cobblemon.mod.common.battles.MoveTarget;
import com.cobblemon.mod.common.battles.ShowdownActionRequest;
import com.cobblemon.mod.common.battles.ShowdownActionResponse;
import com.cobblemon.mod.common.battles.ShowdownMoveset;
import com.cobblemon.mod.common.battles.SwitchActionResponse;
import com.cobblemon.mod.common.battles.Targetable;
import kotlin.jvm.functions.Function1;
import com.cobblemon.mod.common.battles.pokemon.BattlePokemon;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** 1단계: 호출되는지와 받는 데이터 형식을 파일로 남기고, 첫 번째 합법 수를 고르는 AI (모델 없음). */
@SuppressWarnings({"rawtypes", "unchecked"})
public class HelloBattleAI implements BattleAI {
    private final PokeAIConfig cfg;
    private final Gson gson = new GsonBuilder().serializeNulls().create();
    private final Map<UUID, Integer> written = new ConcurrentHashMap<>();  // 배틀별로 이미 기록한 프로토콜 줄 수

    public HelloBattleAI(PokeAIConfig cfg) {
        this.cfg = cfg;
    }

    protected void log(UUID battleId, String text) {
        try {
            Path dir = Path.of("pokeai-log");
            Files.createDirectories(dir);
            Files.writeString(dir.resolve(battleId + ".log"), text + "\n", StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (Exception e) {
            PokeAIMod.LOG.warn("[pokeai] log write failed: {}", e.toString());
        }
    }

    protected String json(Object o) {
        try {
            return gson.toJson(o);
        } catch (Throwable t) {  // Kotlin lazy 위임 필드 등에서 직렬화가 막히면 문자열로
            return "<gson 실패: " + t + "> " + o;
        }
    }

    @Override
    public ShowdownActionResponse choose(ActiveBattlePokemon active, PokemonBattle battle, BattleSide aiSide,
                                         ShowdownMoveset moveset, boolean forceSwitch) {
        UUID id = battle.getBattleId();
        BattleActor actor = active.getActor();
        try {
            log(id, "==== choose() turn=" + battle.getTurn() + " forceSwitch=" + forceSwitch + " actor=" + actor.getShowdownId()
                    + " active=" + (active.getBattlePokemon() == null ? "null" : active.getBattlePokemon().getUuid()));
            if (moveset != null) {
                StringBuilder sb = new StringBuilder("moveset: trapped=" + moveset.getTrapped() + " tera=" + moveset.getCanTerastallize() + " moves=");
                for (InBattleMove m : moveset.getMoves()) sb.append(m.getId()).append("(pp ").append(m.getPp()).append(m.getDisabled() ? ",disabled" : "").append(") ");
                log(id, sb.toString());
            }
            ShowdownActionRequest req = actor.getRequest();
            log(id, "request(actor.getRequest) = " + json(req));
            // 원본 프로토콜: 이번 호출 전까지 새로 쌓인 줄만 추가로 기록
            List<String> msgs = battle.getShowdownMessages();
            int from = written.getOrDefault(id, 0);
            int to = msgs.size();
            for (int i = from; i < to; i++) log(id, "msg[" + i + "] " + msgs.get(i).replace("\n", "\n"));
            written.put(id, to);
        } catch (Throwable t) {
            log(id, "!! 데이터 기록 중 예외: " + t);
        }

        ShowdownActionResponse r = firstValid(active, actor, moveset, forceSwitch, id);
        if (r != null) return r;
        log(id, "-> default");
        return new DefaultActionResponse();
    }

    /** 클라이언트가 실제로 대상을 고를 수 있는 기술 타입인지. 그 외(자신/전체 대상, 쇼다운이 자동으로 정하는
     *  randomNormal/scripted 등)에 명시적으로 대상을 붙이면 "Can't choose a target" 오류로 배틀이 멈춤. */
    protected static boolean isClientChosenTarget(MoveTarget t) {
        return t == MoveTarget.normal || t == MoveTarget.any || t == MoveTarget.adjacentFoe
                || t == MoveTarget.adjacentAlly || t == MoveTarget.adjacentAllyOrSelf;
    }

    /** 사용 가능한 첫 번째 합법 행동(기술 → 교체). 브리지 실패 시의 안전한 대체 행동으로도 씀. */
    protected ShowdownActionResponse firstValid(ActiveBattlePokemon active, BattleActor actor, ShowdownMoveset moveset,
                                                boolean forceSwitch, UUID id) {
        try {
            if (!forceSwitch && moveset != null) {
                for (InBattleMove m : moveset.getMoves()) {
                    // 기술 대상(pnx)이 반드시 필요함: 기술 종류별 유효 대상 목록의 첫 번째, 없으면 정면 상대
                    String pnx = null;
                    MoveTarget mt = m.getTarget();
                    // 클라이언트가 실제로 고를 수 있는 대상 타입일 때만 계산 → 그 외(자신/전체/자동 대상)는 명시하면 Invalid choice로 배틀 멈춤
                    if (isClientChosenTarget(mt)) {
                        try {
                            Function1 f = mt.getTargetList();
                            Object r = f.invoke(active);
                            if (r instanceof List && !((List) r).isEmpty()) pnx = ((Targetable) ((List) r).get(0)).getPNX();
                        } catch (Throwable t) {
                            log(id, "  (대상 목록 계산 실패: " + t + ")");
                        }
                        if (pnx == null) pnx = active.getOppositeOpponent().getPNX();
                    }
                    ShowdownActionResponse resp = new MoveActionResponse(m.getId(), pnx, null);
                    // Cobblemon의 isValid()는 self/all류에도 targetPnx가 유효 대상 목록에 있어야 한다고 오판함 → canBeUsed()로 직접 판정
                    boolean ok = isClientChosenTarget(mt) ? resp.isValid(active, moveset, forceSwitch) : m.canBeUsed();
                    log(id, "  후보 move " + m.getId() + " target=" + pnx + " valid=" + ok);
                    if (ok) {
                        log(id, "-> move " + m.getId() + " " + pnx);
                        return resp;
                    }
                }
            }
            for (BattlePokemon p : actor.getPokemonList()) {
                if (p.getHealth() > 0 && !p.isSentOut()) {
                    ShowdownActionResponse resp = new SwitchActionResponse(p.getUuid());
                    boolean ok = resp.isValid(active, moveset, forceSwitch);
                    log(id, "  후보 switch " + p.getUuid() + " valid=" + ok);
                    if (ok) {
                        log(id, "-> switch " + p.getUuid());
                        return resp;
                    }
                }
            }
        } catch (Throwable t) {
            log(id, "!! 행동 선택 중 예외: " + t);
        }
        return null;
    }
}
