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
    private final ObjectMapper objectMapper; // DI를 통한 주입

    @Value("${kafka.topic.news-ai}")
    private String newsAiTopic;

    @KafkaListener(topics = "${kafka.topic.news-crawl}", groupId = "${spring.kafka.consumer.group-id}", concurrency = "5")
    public void listen(String message) {
        log.info("Kafka 메시지 수신: {}", message);
        try {
            // 크롤링 요청 메시지를 파싱합니다.
            NewsCrawlerRequest newsCrawlerRequest = objectMapper.readValue(message, NewsCrawlerRequest.class);
            String stockName = newsCrawlerRequest.getStockName();

            // 뉴스 크롤링을 수행하여 뉴스 아이템 리스트를 얻습니다.
            List<NewsItem> newsItemList = newsCrawlerService.fetchNews(newsCrawlerRequest);

            // 리스트가 비어있으면 메시지 전송을 생략합니다.
            if (newsItemList == null || newsItemList.isEmpty()) {
                log.info("크롤링 결과 뉴스 아이템이 없습니다. 메시지 전송을 생략합니다. (종목: {})", stockName);
                return;
            }

            // 뉴스 크롤링 결과를 AI 분석 단계로 전달하기 위해 새로운 메시지 생성
            String aiMessage = objectMapper.writeValueAsString(NewsAiRequest.of(stockName, newsItemList));

            // 새로운 Kafka 토픽으로 메시지를 전송합니다.
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
