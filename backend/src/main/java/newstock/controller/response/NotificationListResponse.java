package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.notification.dto.NewsNotificationDto;
import newstock.domain.notification.dto.NotificationDto;
import newstock.domain.notification.dto.StockNotificationDto;

import java.util.List;

@Getter
@Builder
public class NotificationListResponse {

    private List<NotificationDto> notificationList;

    private NewsNotificationDto newsInfo;

    private StockNotificationDto stockInfo;


    public static NotificationListResponse of(List<NotificationDto> notificationList,
                                              NewsNotificationDto newsInfo,
                                              StockNotificationDto stockInfo) {
        return NotificationListResponse.builder()
                .notificationList(notificationList)
                .newsInfo(newsInfo)
                .stockInfo(stockInfo)
                .build();
    }

}
