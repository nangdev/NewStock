package newstock.domain.stock.dto;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.stock.entity.Stock;

import java.io.File;

@Builder
@Getter
public class StockDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

    private File stockImage;

    public static StockDto of(Stock stock) {
        return StockDto.builder()
                .stockId(stock.getStockId())
                .stockCode(stock.getStockCode())
                .stockName(stock.getStockName())
                .stockImage(new File(stock.getImgUrl()))
                .build();
    }

}
