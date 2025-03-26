package newstock.domain.stock.dto;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.stock.entity.Stock;

import java.io.File;

@Builder
@Getter
public class StockDto {
    private Integer stockCode;
    private String stockName;
    private File stockImage;

    public static StockDto fromStock(Stock stock) {
        return StockDto.builder()
                .stockCode(stock.getStockCode())
                .build();
    }
}
