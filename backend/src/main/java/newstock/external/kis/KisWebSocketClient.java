package newstock.external.kis;

import jakarta.websocket.*;
import lombok.extern.slf4j.Slf4j;
import java.net.URI;

@ClientEndpoint
@Slf4j
public class KisWebSocketClient {
    private Session session;

    public boolean isConnected() {
        return session != null && session.isOpen();
    }

    public void connect() throws Exception {
        try {
            URI endpointURI = new URI("ws://ops.koreainvestment.com:21000");
            WebSocketContainer container = ContainerProvider.getWebSocketContainer();
            container.connectToServer(this, endpointURI);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @OnOpen
    public void onOpen(Session session) {
        this.session = session;
        log.info("한투 웹소켓 연결 성공");
    }

    @OnMessage
    public void onMessage(String message) {
        log.info(message);
    }

    @OnClose
    public void onClose(Session session) {
        this.session = null;
        log.info("한투 웹소켓 연결 종료");
    }

}
