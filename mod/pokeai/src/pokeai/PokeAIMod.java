package pokeai;

import com.gitlab.srcmc.rctapi.api.util.JTO;
import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PokeAIMod implements ModInitializer {
    public static final Logger LOG = LoggerFactory.getLogger("pokeai");

    @Override
    public void onInitialize() {
        // RCT 트레이너 JSON의 "ai.type" 이름 "pokeai"를 우리 BattleAI로 연결
        JTO.registerParser("pokeai", (PokeAIConfig cfg) -> new BridgeBattleAI(cfg), PokeAIConfig::new, PokeAIConfig.class);
        JTO.registerParser("pokeai_hello", (PokeAIConfig cfg) -> new HelloBattleAI(cfg), PokeAIConfig::new, PokeAIConfig.class);
        LOG.info("[pokeai] AI types 'pokeai' (bridge) and 'pokeai_hello' registered");
    }
}
