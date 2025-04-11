package newstock.kafka.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.notification.dto.NotificationDto;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NotificationRequest {

    private List<NotificationDto> notifications;

    public static NotificationRequest of(List<NotificationDto> notifications) {
        return NotificationRequest.builder()
                .notifications(notifications)
                .build();
    }
}
