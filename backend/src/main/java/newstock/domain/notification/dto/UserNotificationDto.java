package newstock.domain.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;


@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserNotificationDto {

    private Integer unId;

    private NewsNotificationDto newsInfo;

    private StockNotificationDto stockInfo;

    private Byte isRead;

}
