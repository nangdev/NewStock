package newstock.external.kis;

import lombok.RequiredArgsConstructor;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class KisWebSocketInitializer {
    private final KisOAuthClient authClient;
    private final KisWebSocketClient webSocketClient;
    private static final String WS_URL = "ws://ops.koreainvestment.com:21000";

    @EventListener(ApplicationReadyEvent.class)
    public void init() {
        if (!webSocketClient.isConnected()) {
            webSocketClient.connect();
        }
    }

}
