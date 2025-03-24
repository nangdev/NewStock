package newstock.domain.news.scheduler;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerScheduler {

    // KafkaTemplate을 주입받아서 메시지 전송에 사용합니다.
    private final KafkaTemplate<String, String> kafkaTemplate;
    private static final String TOPIC = "news-crawl-topic";

    // 크롤링할 종목 리스트 정의
    private final List<String> stockNames = List.of("삼성전자", "현대자동차", "LG전자", "네이버", "카카오");

    // 매 분 0초마다 실행 (cron expression: "0 * * * * *")
    @Scheduled(cron = "0 * * * * *")
    public void scheduleNewsCrawling() {
        for (String stock : stockNames) {
            try {
                // 각 종목명을 Kafka 토픽으로 전송
                kafkaTemplate.send(TOPIC, stock);
                log.info("Kafka 메시지 전송 완료: {}", stock);
            } catch (Exception e) {
                log.error("Kafka 메시지 전송 중 오류 발생 ({}): {}", stock, e.getMessage());
            }
        }
    }
}
