package newstock.kafka.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.notification.dto.NotificationDto;
import newstock.domain.notification.service.NotificationService;
import newstock.kafka.request.NotificationRequest;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class NotificationConsumer {

    private final NotificationService notificationService;
    private final ObjectMapper objectMapper;

    @KafkaListener(topics = "${kafka.topic.news-notification}", groupId = "${kafka.consumer.group.news-notification}")
    public void listen(String message) {
        log.info("Kafka 푸쉬 알림 메시지 수신 : {}", message);
        try {
            NotificationRequest req = objectMapper.readValue(message, NotificationRequest.class);

            for(NotificationDto dto : req.getNotifications()) {
                notificationService.addNotifications(dto);
            }

        } catch (Exception e) {
            log.error("Notification 컨슈머 에러 발생 : {}", e.getMessage());
        }
    }

}
