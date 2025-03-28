package newstock.domain.news.scheduler;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.service.StockService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerScheduler {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final StockService stockService;

    @Value("${kafka.topic.news-crawl}")
    private String topic;

    // 스케줄러가 매 분 0초마다 실행됩니다.
    @Scheduled(cron = "0 * * * * *")
    public void scheduleNewsCrawling() {
        List<StockDto> stockList = stockService.getAllStocks();
        Instant schedulerTime = Instant.now();

        for (StockDto stock : stockList) {
            try {
                String message = String.format(
                        "{\"stockName\":\"%s\", \"stockId\":\"%s\", \"schedulerTime\":\"%s\"}",
                        stock.getStockName(), stock.getStockId(), schedulerTime.toString());

                kafkaTemplate.send(topic, String.valueOf(stock.getStockId()), message);
                log.info("Kafka 메시지 전송 완료: {}", message);
            } catch (Exception e) {
                log.error("Kafka 메시지 전송 중 오류 발생 ({}): {}", stock.getStockName(), e.getMessage());
            }
        }
    }
}
