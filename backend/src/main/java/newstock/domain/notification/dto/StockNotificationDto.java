package newstock.domain.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.stock.entity.Stock;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StockNotificationDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

    public static StockNotificationDto of(Stock stock) {
        return StockNotificationDto.builder()
                .stockId(stock.getStockId())
                .stockCode(stock.getStockCode())
                .stockName(stock.getStockName())
                .build();
    }

}
