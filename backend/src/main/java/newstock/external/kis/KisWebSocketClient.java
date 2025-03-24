package newstock.external.kis;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.websocket.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.List;

@ClientEndpoint
@Slf4j
@RequiredArgsConstructor
public class KisWebSocketClient {
    private Session session;

    private final KisOAuthClient authClient;
    private final ObjectMapper objectMapper;

    public boolean isConnected() {
        return session != null && session.isOpen();
    }

    public void connect() throws Exception {
        try {
            URI endpointURI = new URI("ws://ops.koreainvestment.com:21000");
            WebSocketContainer container = ContainerProvider.getWebSocketContainer();
            container.connectToServer(this, endpointURI);


        } catch (Exception e) {

        }
    }

    @OnOpen
    public void onOpen(Session session) {
        this.session = session;
        log.info("한투 웹소켓 연결 성공");

        RemoteEndpoint.Basic remote = session.getBasicRemote();


        try {
            String approvalKey = authClient.getWebSocketKey().getApprovalKey();

            KisHeaderDto header = new KisHeaderDto();
            header.setApprovalKey(approvalKey);
            header.setCusttype("P");
            header.setTrType("1");
            header.setContentType("utf-8");

            List<String> codes = Arrays.stream(KospiStock.values())
                    .map(KospiStock::getCode)
                    .toList();

            for (String code: codes) {

                KisWebSocketSubMsg msg = new KisWebSocketSubMsg();
                KisBodyDto body = new KisBodyDto();
                KisInputDto inputDto = new KisInputDto();

                inputDto.setTrId("H0STCNT0");
                inputDto.setTrKey(code);
                body.setInput(inputDto);

                msg.setHeader(header);
                msg.setBody(body);
                String jsonString = objectMapper.writeValueAsString(msg);

                remote.sendText(jsonString);
                log.info("subscribe {}", code);
            }


        } catch (IOException ioe) {

        } catch (Exception e) {

        }


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
