package newstock.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.AnalysisRequest;
import newstock.domain.news.dto.AnalysisResponse;
import newstock.domain.news.dto.NewsAiRequest;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.dto.SummarizationRequest;
import newstock.domain.news.dto.SummarizationResponse;
import newstock.domain.news.service.NewsAiService;
import newstock.domain.news.service.NewsService;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsAiConsumer {

    private final NewsAiService newsAiService;
    private final NewsService newsService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @KafkaListener(topics = "${kafka.topic.news-ai}", groupId = "${kafka.consumer.group.news-ai}")
    public void listen(String message) {
        log.info("Kafka AI 메시지 수신: {}", message);
        try {
            // Kafka 메시지를 NewsAiRequest 객체로 역직렬화합니다.
            NewsAiRequest aiRequest = objectMapper.readValue(message, NewsAiRequest.class);
            String stockName = aiRequest.getStockName();
            List<NewsItem> newsItemList = aiRequest.getNewsItemList();
            List<NewsItem> filteredNewsItems = new ArrayList<>();

            // 각 뉴스 아이템에 대해 AI 분석 및 요약 작업 수행
            for (NewsItem item : newsItemList) {
                AnalysisResponse analysisResponse = newsAiService.analysis(
                        AnalysisRequest.of(item.getTitle(), item.getContent())
                );
                log.info("점수 채점 완료! 점수: {}", analysisResponse.getScore());
                // 점수가 조건에 부합하지 않으면 건너뜁니다.
                if (!(analysisResponse.getScore() > 5 || analysisResponse.getScore() < -5)) {
                    continue;
                }
                item.setScore(analysisResponse.getScore());
                // 분석 후 AI가 반환한 내용을 업데이트
                item.setContent(analysisResponse.getContent());
                try {
                    SummarizationResponse summarizationResponse = newsAiService.summarize(
                            SummarizationRequest.of(item.getContent(), 300, 40, false)
                    );
                    item.setNewsSummary(summarizationResponse.getSummaryContent());
                } catch (Exception e) {
                    item.setNewsSummary("");
                }
                filteredNewsItems.add(item);
            }
            // 최종 분석된 뉴스 아이템들을 DB에 저장하거나 후속 처리합니다.
            newsService.addNewsItems(filteredNewsItems);
            log.info("AI 분석 및 요약 완료, 종목: {} 뉴스 개수: {}", stockName, filteredNewsItems.size());
        } catch (Exception e) {
            log.error("뉴스 AI 분석 중 오류 발생: {}", e.getMessage());
        }
    }
}
