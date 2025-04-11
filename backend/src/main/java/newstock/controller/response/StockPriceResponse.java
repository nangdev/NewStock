package newstock.controller.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.stockprice.dto.StockPriceDto;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StockPriceResponse {

    List<StockPriceDto> stockPrices;

    public static StockPriceResponse of(List<StockPriceDto> stockPrices) {
        return StockPriceResponse.builder()
                .stockPrices(stockPrices)
                .build();
    }
}
