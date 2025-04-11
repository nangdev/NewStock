package newstock.kafka.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.AnalysisRequest;
import newstock.domain.news.dto.AnalysisResponse;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.dto.SummarizationRequest;
import newstock.domain.news.dto.SummarizationResponse;
import newstock.domain.news.service.NewsAiService;
import newstock.kafka.request.NewsAiRequest;
import newstock.kafka.request.NewsDbRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsAiConsumer {

    private final NewsAiService newsAiService;
    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${kafka.topic.news-db}")
    private String newsDbTopic;

    @KafkaListener(topics = "${kafka.topic.news-ai}", groupId = "${kafka.consumer.group.news-ai}", concurrency = "1")
    public void listen(String message) {
        log.info("Kafka AI 분석 메시지 수신");
        try {
            NewsAiRequest aiRequest = objectMapper.readValue(message, NewsAiRequest.class);
            String stockName = aiRequest.getStockName();
            List<NewsItem> newsItemList = aiRequest.getNewsItemList();
            List<NewsItem> filteredNewsItems = new ArrayList<>();

            for (NewsItem item : newsItemList) {
                AnalysisResponse analysisResponse = newsAiService.analysis(
                        AnalysisRequest.of(item.getTitle(), item.getContent())
                );
                if (!(analysisResponse.getScore() > 2 || analysisResponse.getScore() < -2)) {
                    continue;
                }
                item.setScores(analysisResponse);
                try {
                    SummarizationResponse summarizationResponse = newsAiService.summarize(
                            SummarizationRequest.of(item.getContent(), 300, 40, false)
                    );
                    item.setNewsSummary(summarizationResponse.getSummaryContent());
                } catch (Exception e) {
                    log.error("요약 처리 중 오류 발생: ", e);
                    item.setNewsSummary("");
                }
                filteredNewsItems.add(item);
            }
            log.info("AI 분석 및 요약 완료, 종목: {} / 뉴스 개수: {}", stockName, filteredNewsItems.size());

            if (filteredNewsItems.isEmpty()) {
                log.info("필터링 결과 뉴스 아이템이 없습니다. DB 저장 메시지 전송을 생략합니다. (종목: {})", stockName);
                return;
            }

            String dbMessage = objectMapper.writeValueAsString(NewsDbRequest.of(stockName, filteredNewsItems));

            kafkaTemplate.send(newsDbTopic, dbMessage)
                    .thenAccept(result -> log.info("Kafka DB 저장 메시지 전송 완료: {}", dbMessage))
                    .exceptionally(ex -> {
                        log.error("Kafka DB 저장 메시지 전송 실패: ", ex);
                        return null;
                    });
        } catch (Exception e) {
            log.error("뉴스 AI 분석 중 오류 발생: ", e);
        }
    }
}
