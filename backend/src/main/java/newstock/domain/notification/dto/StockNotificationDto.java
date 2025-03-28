package newstock.domain.notification.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class StockNotificationDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

}
