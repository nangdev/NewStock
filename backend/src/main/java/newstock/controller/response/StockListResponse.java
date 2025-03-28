package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.stock.dto.StockDto;

import java.util.List;

@Builder
@Getter
public class StockListResponse {

    private List<StockDto> stockList;

    public static StockListResponse of(List<StockDto> stockList) {
        return StockListResponse.builder()
                .stockList(stockList)
                .build();
    }

}
