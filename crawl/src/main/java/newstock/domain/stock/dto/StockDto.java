package newstock.domain.stock.dto;

import lombok.Builder;
import lombok.Data;
import newstock.domain.stock.entity.Stock;

@Data
@Builder
public class StockDto {

    private Integer stockId;

    private String stockName;

    public static StockDto of(Stock stock) {
        return StockDto.builder()
                .stockId(stock.getStockId())
                .stockName(stock.getStockName())
                .build();
    }
}
