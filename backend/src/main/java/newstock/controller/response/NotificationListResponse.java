package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.notification.dto.UserNotificationDto;

import java.util.List;

@Getter
@Builder
public class NotificationListResponse {

    private List<UserNotificationDto> notificationList;

    public static NotificationListResponse of(List<UserNotificationDto> notificationList) {
        return NotificationListResponse.builder()
                .notificationList(notificationList)
                .build();
    }

}
