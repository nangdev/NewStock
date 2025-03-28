package newstock.domain.notification.dto;

import lombok.Builder;
import lombok.Data;


@Data
@Builder
public class UserNotificationDto {

    private Integer unId;

    private NewsNotificationDto newsInfo;

    private StockNotificationDto stockInfo;

    private Byte isRead;

}
