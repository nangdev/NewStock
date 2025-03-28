package newstock.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.*;
import newstock.domain.news.service.NewsCrawlerService;
import newstock.domain.news.service.NewsService;
import newstock.domain.news.service.NewsAiService;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerConsumer {

    private final NewsCrawlerService newsCrawlerService;
    private final NewsService newsService;
    private final NewsAiService newsAiService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @KafkaListener(topics = "news-crawl-topic", groupId = "news-crawl-group")
    public void listen(String message) {
        log.info("Kafka 메시지 수신: {}", message);
        try {
            StockMessage stockMessage = objectMapper.readValue(message, StockMessage.class);
            String stockName = stockMessage.getStockName();

            List<NewsItem> newsItemList = newsCrawlerService.fetchNews(stockMessage);
            List<NewsItem> filteredNewsItems = new ArrayList<>();

            for (NewsItem item : newsItemList) {
                AnalysisResponse analysisResponse = newsAiService.analysis(AnalysisRequest.of(item.getContent()));
                // 점수가 조건에 부합하지 않으면 바로 다음 항목으로 넘어감
                log.info("점수 채점 완료! 점수 : {}",analysisResponse.getScore());
                if (!(analysisResponse.getScore() > 5 || analysisResponse.getScore() < -5)) {
                    continue;
                }
                item.setScore(analysisResponse.getScore());
                // 조건에 맞는 경우 처리
                item.setContent(analysisResponse.getContent());
                try {
                    SummarizationResponse summarizationResponse = newsAiService.summarize(item.getContent(), 300, 40, false);
                    item.setNewsSummary(summarizationResponse.getSummaryContent());
                } catch (Exception e) {
                    item.setNewsSummary("");
                }
                filteredNewsItems.add(item);
            }
            newsService.addNewsItems(filteredNewsItems);
            log.info("크롤링 및 요약 완료, 종목: {} 뉴스 개수: {}", stockName, filteredNewsItems.size());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("뉴스 크롤링 인터럽트 발생: {}", e.getMessage());
        } catch (Exception e) {
            log.error("뉴스 크롤링 중 오류 발생: {}", e.getMessage());
        }
    }

}
