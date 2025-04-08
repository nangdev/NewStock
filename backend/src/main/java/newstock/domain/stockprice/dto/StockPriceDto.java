package newstock.domain.stockprice.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.stockprice.entity.StockPrice;

import java.math.BigDecimal;
import java.time.LocalDate;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StockPriceDto {

    private Integer stockId;

    private LocalDate date;

    private int price;

    public static StockPriceDto of(StockPrice stockPrice) {
        return StockPriceDto.builder()
                .stockId(stockPrice.getStockId())
                .date(stockPrice.getDate())
                .price(stockPrice.getPrice())
                .build();
    }

    public StockPrice toEntity() {
        return StockPrice.builder()
                .stockId(this.stockId)
                .date(this.date)
                .price(this.price)
                .build();
    }
}
