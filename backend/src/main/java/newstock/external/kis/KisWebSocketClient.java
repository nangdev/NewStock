package newstock.external.kis;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.websocket.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.stock.service.StockPriceService;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ExternalApiException;
import newstock.external.kis.dto.KisBodyDto;
import newstock.external.kis.dto.KisHeaderDto;
import newstock.external.kis.dto.KisInputDto;
import newstock.external.kis.dto.KisStockInfoDto;

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

    public void connect() {
        try {
            URI endpointURI = new URI("ws://ops.koreainvestment.com:21000");
            WebSocketContainer container = ContainerProvider.getWebSocketContainer();
            container.connectToServer(this, endpointURI);
        } catch (Exception e) {
            log.error("한투 웹소켓 연결 실패");
            throw new ExternalApiException(ExceptionCode.EXTERNAL_API_ERROR);
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

            for (String code : codes) {
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

        } catch (IllegalArgumentException | IOException ioe) {
            log.error("한투 웹소켓 구독 오류");
            throw new ExternalApiException(ExceptionCode.EXTERNAL_API_ERROR);
        } catch (Exception e) {
            log.error("한투 웹소켓 세션 오류");
            throw new ExternalApiException(ExceptionCode.BUSINESS_ERROR);
        }
    }

    @OnMessage
    public void onMessage(String message) {
        log.info(message);
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
                // 1페이지만 사용
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
            // 에러 무시하고 웹소켓 연결 유지
            log.error("한투 웹소켓 파싱 에러");
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
