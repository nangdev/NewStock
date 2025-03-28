package newstock.kafka.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.service.NewsService;
import newstock.kafka.request.NewsDbRequest;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsDbConsumer {

    private final NewsService newsService;
    private final ObjectMapper objectMapper; // 생성자 주입 방식으로 DI

    @KafkaListener(topics = "${kafka.topic.news-db}", groupId = "${kafka.consumer.group.news-db}")
    public void listen(String message) {
        log.info("Kafka DB 저장 메시지 수신: {}", message);
        try {
            // 메시지를 NewsDbRequest 객체로 역직렬화
            NewsDbRequest dbRequest = objectMapper.readValue(message, NewsDbRequest.class);
            String stockName = dbRequest.getStockName();

            // DB 저장 로직 수행 (DB 저장 성공 시 후속 이벤트 발행 등 추가 처리는 여기서 진행)
            newsService.addNewsItems(dbRequest.getFilteredNewsItems());
            log.info("DB 저장 완료, 종목: {} / 뉴스 개수: {}", stockName, dbRequest.getFilteredNewsItems().size());
        } catch (Exception e) {
            log.error("뉴스 DB 저장 중 오류 발생: ", e);
        }
    }
}
