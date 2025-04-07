package newstock.domain.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StockNotificationDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

}
