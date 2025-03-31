package newstock.domain.news.scheduler;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.service.StockService;
import newstock.kafka.request.NewsCrawlerRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerScheduler {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final StockService stockService;
    private final ObjectMapper objectMapper;

    @Value("${kafka.topic.news-crawl}")
    private String topic;

    // 단일 스레드 Executor를 사용해 스케줄 작업들을 순차적으로 실행합니다.
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    // 스케줄러가 매 분 0초마다 실행됩니다.
//    @Scheduled(cron = "0 * * * * *")
    public void scheduleNewsCrawling() {
        // 작업을 단일 스레드 Executor에 제출하여, 이전 작업이 끝날 때까지 대기하도록 합니다.
        executor.submit(() -> {
            try {
                List<StockDto> stockList = stockService.getAllStocks();
                Instant schedulerTime = Instant.now();

                // 종목 목록을 병렬 스트림으로 처리하여 동시에 메시지를 전송합니다.
                stockList.parallelStream().forEach(stock -> {
                    try {
                        String message = objectMapper.writeValueAsString(
                                NewsCrawlerRequest.of(stock.getStockName(), stock.getStockId(), schedulerTime.toString())
                        );
                        CompletableFuture<SendResult<String, String>> future =
                                kafkaTemplate.send(topic, String.valueOf(stock.getStockId()), message);
                        future.thenAccept(sendResult ->
                                log.info("Kafka 메시지 전송 성공: {}", message)
                        ).exceptionally(ex -> {
                            log.error("Kafka 메시지 전송 실패 ({}): ", stock.getStockName(), ex);
                            return null;
                        });
                    } catch (Exception e) {
                        log.error("메시지 직렬화 중 오류 발생 ({}): ", stock.getStockName(), e);
                    }
                });
            } catch (Exception e) {
                log.error("스케줄 작업 실행 중 예외 발생: ", e);
            }
        });
    }

    @PreDestroy
    public void shutdownExecutor() {
        executor.shutdown();
    }
}
