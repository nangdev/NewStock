package newstock.domain.news.scheduler;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.enums.KospiStock;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerScheduler {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private static final String TOPIC = "news-crawl-topic";

    // 스케줄러가 매 분 0초마다 실행됩니다.
    // 여기서는 각 종목에 대해 stockName, stockCode를 전송합니다.
    @Scheduled(cron = "0 * * * * *")
    public void scheduleNewsCrawling() {
        for (KospiStock stock : KospiStock.values()) {
            try {
                String message = String.format(
                        "{\"stockName\":\"%s\", \"stockCode\":\"%s\"}",
                        stock.getName(), stock.getCode());

                // stock.getCode()를 key로 지정하여 메시지가 파티셔닝되도록 합니다.
                kafkaTemplate.send(TOPIC, stock.getCode(), message);
                log.info("Kafka 메시지 전송 완료: {}", message);
            } catch (Exception e) {
                log.error("Kafka 메시지 전송 중 오류 발생 ({}): {}", stock.getName(), e.getMessage());
            }
        }
    }
}
