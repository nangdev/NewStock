package newstock.controller.request;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class StockNewsRequest {

    private int stockCode;

    private int page;

    private int count;

    private String sort;

    public static StockNewsRequest of(int stockCode, int page, int count, String sort) {
        return StockNewsRequest.builder()
                .stockCode(stockCode)
                .page(page)
                .count(count)
                .sort(sort)
                .build();
    }


}
