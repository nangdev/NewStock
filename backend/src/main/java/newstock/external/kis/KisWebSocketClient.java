package newstock.external.kis;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.websocket.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.stock.service.StockPriceService;

import java.io.IOException;
import java.net.URI;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;

@ClientEndpoint
@Slf4j
@RequiredArgsConstructor
public class KisWebSocketClient {
    private Session session;
    private RemoteEndpoint.Basic remote = null;
    private final KisOAuthClient authClient;
    private final ObjectMapper objectMapper;
    private final StockPriceService stockPriceService;

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

        this.remote = session.getBasicRemote();

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
        // 파싱
        try {
            // 구독 성공, pingpong -> json 형태
            if (isJson(message)) {
                JsonNode node = objectMapper.readTree(message);
                // pinppong
                if (node.get("header").get("tr_id").asText().equals("PINGPONG")) {
                    remote.sendText(message);   // 그대로 응답
                    log.info(message);
                }
            }
            // 주가 데이터
            else {
                String[] data = message.split("\\|");
                // 맨 앞자리가 1이면 암호화되어있는데 단순 체결가는 복호화가 필요없다.
                // 한번에 여러 데이터가 들어오면 페이징되어서 데이터가 들어오는데 1페이지만 사용
                boolean isEncoded = data[0].equals("1");
                int pages = Integer.parseInt(data[2]);

                if (!isEncoded) {
                    String[] parsedData = data[3].split("\\^");
                    String stockCode = parsedData[0];
                    DateTimeFormatter inputFormatter = DateTimeFormatter.ofPattern("HHmmss");
                    LocalTime time = LocalTime.parse(parsedData[1], inputFormatter);
                    int price = Integer.parseInt(parsedData[2]);
                    double changeRate = Double.parseDouble(parsedData[5]);

                    KisStockInfoDto stockInfoDto = KisStockInfoDto.builder()
                            .stockCode(stockCode)
                            .time(time)
                            .price(price)
                            .changeRate(changeRate)
                            .build();

                    // websocket 브로커
                    stockPriceService.sendStockInfo(stockInfoDto);
                }

            }
        } catch (Exception e) {

        }


    }

    @OnClose
    public void onClose(Session session) {
        this.session = null;
        log.info("한투 웹소켓 연결 종료");
    }

    private boolean isJson(String message) {
        return message.trim().startsWith("{");
    }

}
