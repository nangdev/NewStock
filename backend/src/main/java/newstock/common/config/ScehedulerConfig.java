package newstock.common.config;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.external.kis.KisWebSocketClient;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;

@Configuration
@EnableScheduling
@RequiredArgsConstructor
@Slf4j
public class ScehedulerConfig {
    private final KisWebSocketClient webSocketClient;

    /**
     * cron: s m h date month week days
     */
    @Scheduled(cron = "0 30 8,12 * * MON-FRI", zone = "Asia/Seoul")
    public void connectKisWebSocket() {
        log.info("한투 웹소켓 스케줄러 동작");
        if (!webSocketClient.isConnected()) {
            log.info("한투 웹소켓 연결 시도");
            webSocketClient.connect();
        } else {
            log.info("한투 웹소켓 연결 확인");
        }
    }
}
