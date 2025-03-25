package newstock.domain.stock.dto;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.stock.entity.Stock;

@Builder
@Getter
public class StockDto {
    private Integer stockCode;
    private String stockName;

    public static StockDto fromStock(Stock stock) {
        return StockDto.builder()
                .stockCode(stock.getStockCode())
                .build();
    }
}
