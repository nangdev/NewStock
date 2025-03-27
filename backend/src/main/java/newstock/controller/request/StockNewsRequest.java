package newstock.controller.request;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class StockNewsRequest {

    private Integer stockId;

    private int page;

    private int count;

    private String sort;

    public static StockNewsRequest of(Integer stockId, int page, int count, String sort) {
        return StockNewsRequest.builder()
                .stockId(stockId)
                .page(page)
                .count(count)
                .sort(sort)
                .build();
    }
}
