package newstock.domain.stock.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.stock.entity.Stock;

@Builder
@NoArgsConstructor
@AllArgsConstructor
@Data
public class StockDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

    private boolean isInterested;

    public static StockDto of(Stock stock) {
        return StockDto.builder()
                .stockId(stock.getStockId())
                .stockCode(stock.getStockCode())
                .stockName(stock.getStockName())
                .build();
    }

}
