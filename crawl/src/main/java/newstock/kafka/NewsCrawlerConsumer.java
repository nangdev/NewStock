package newstock.kafka;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NewsItem;
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

    // Kafka 토픽 "news-crawl-topic"에 메시지가 오면 실행됩니다.
    @KafkaListener(topics = "news-crawl-topic", groupId = "news-crawl-group")
    public void listen(String stockName) {
        log.info("Kafka 메시지 수신: {}", stockName);
        try {
            // 각 종목명에 대해 크롤링 작업 수행
            List<NewsItem> newsItemList = newsCrawlerService.fetchNews(stockName);
            newsService.saveNewsItems(newsItemList);
            // 여기서 수집된 뉴스들을 DB에 저장하거나 후속 처리를 진행할 수 있습니다.
            log.info("크롤링 완료, 종목: {} 뉴스 개수: {}", stockName, newsItemList.size());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("뉴스 크롤링 인터럽트 발생 ({}): {}", stockName, e.getMessage());
        } catch (Exception e) {
            log.error("뉴스 크롤링 중 오류 발생 ({}): {}", stockName, e.getMessage());
        }
    }
}
