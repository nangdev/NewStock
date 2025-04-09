package newstock.kafka.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.service.NewsService;
import newstock.domain.notification.dto.NotificationDto;
import newstock.kafka.request.NewsDbRequest;
import newstock.kafka.request.NotificationRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsDbConsumer {

    private final NewsService newsService;
    private final ObjectMapper objectMapper;
    private final KafkaTemplate<String, String> kafkaTemplate;

    @Value("${kafka.topic.news-notification}")
    private String notificationTopic;

    @KafkaListener(topics = "${kafka.topic.news-db}", groupId = "${kafka.consumer.group.news-db}", concurrency = "1")
    public void listen(String message) {
        log.info("Kafka DB 저장 메시지 수신");
        try {
            NewsDbRequest dbRequest = objectMapper.readValue(message, NewsDbRequest.class);
            String stockName = dbRequest.getStockName();

            List<NewsItem> newsItems = dbRequest.getFilteredNewsItems();
            newsService.addNewsItems(newsItems);
            log.info("DB 저장 완료, 종목: {} / 뉴스 개수: {}", stockName, dbRequest.getFilteredNewsItems().size());

            List<NotificationDto> scoreFilteredNotificationDtos = newsItems.stream()
                    .filter(newsItem -> newsItem.getScore() > 7.8)
                    .map(NotificationDto::of)
                    .toList();

            if(!scoreFilteredNotificationDtos.isEmpty()) {
                String notificationMessage = objectMapper.writeValueAsString(NotificationRequest.of(scoreFilteredNotificationDtos));

                kafkaTemplate.send(notificationTopic, notificationMessage)
                        .thenAccept(result -> log.info("Kafka 푸쉬 알림 메시지 전송 완료: {}", notificationMessage))
                        .exceptionally(ex -> {
                            log.error("Kafka 푸쉬 알림 메시지 전송 실패: ", ex);
                            return null;
                        });
            }

        } catch (Exception e) {
            log.error("뉴스 DB 저장 중 오류 발생: ", e);
        }
    }
}
