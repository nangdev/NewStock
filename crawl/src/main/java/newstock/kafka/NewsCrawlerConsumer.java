package newstock.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.dto.StockMessage;
import newstock.domain.news.service.NewsCrawlerService;
import newstock.domain.news.service.NewsService;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerConsumer {

    private final NewsCrawlerService newsCrawlerService;
    private final NewsService newsService;
    // Jackson의 ObjectMapper를 이용해 JSON 메시지를 파싱합니다.
    private final ObjectMapper objectMapper = new ObjectMapper();

    // Kafka 토픽 "news-crawl-topic"에서 메시지를 수신하면 실행됩니다.
    @KafkaListener(topics = "news-crawl-topic", groupId = "news-crawl-group")
    public void listen(String message) {
        log.info("Kafka 메시지 수신: {}", message);
        try {
            StockMessage stockMessage = objectMapper.readValue(message, StockMessage.class);
            String stockName = stockMessage.getStockName();

            // 각 종목명에 대해 크롤링 작업 수행
            List<NewsItem> newsItemList = newsCrawlerService.fetchNews(stockMessage);
            newsService.saveNewsItems(newsItemList);
            log.info("크롤링 완료, 종목: {} 뉴스 개수: {}", stockName, newsItemList.size());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("뉴스 크롤링 인터럽트 발생: {}", e.getMessage());
        } catch (Exception e) {
            log.error("뉴스 크롤링 중 오류 발생: {}", e.getMessage());
        }
    }
}
