package newstock.kafka.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.kafka.request.NewsAiRequest;
import newstock.kafka.request.NewsCrawlerRequest;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.service.NewsCrawlerService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerConsumer {

    private final NewsCrawlerService newsCrawlerService;
    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${kafka.topic.news-ai}")
    private String newsAiTopic;

    @KafkaListener(topics = "${kafka.topic.news-crawl}", groupId = "${spring.kafka.consumer.group-id}", concurrency = "5")
    public void listen(String message) {
        log.info("Kafka 메시지 수신: {}", message);
        try {
            NewsCrawlerRequest newsCrawlerRequest = objectMapper.readValue(message, NewsCrawlerRequest.class);
            String stockName = newsCrawlerRequest.getStockName();

            List<NewsItem> newsItemList = newsCrawlerService.fetchNews(newsCrawlerRequest);

            if (newsItemList == null || newsItemList.isEmpty()) {
                log.info("크롤링 결과 뉴스 아이템이 없습니다. 메시지 전송을 생략합니다. (종목: {})", stockName);
                return;
            }

            String aiMessage = objectMapper.writeValueAsString(NewsAiRequest.of(stockName, newsItemList));

            kafkaTemplate.send(newsAiTopic, aiMessage)
                    .thenAccept(result -> log.info("Kafka AI 분석 메시지 전송 완료: {}", aiMessage))
                    .exceptionally(ex -> {
                        log.error("Kafka AI 분석 메시지 전송 실패: ", ex);
                        return null;
                    });
        } catch (Exception e) {
            log.error("뉴스 크롤링 중 오류 발생: ", e);
        }
    }
}
