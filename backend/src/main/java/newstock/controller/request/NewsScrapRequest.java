package newstock.controller.request;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class NewsScrapRequest {

    private int stockCode;

    private int page;

    private int count;

    private String sort;

    private int userId;

    public static NewsScrapRequest of(int stockCode, int page, int count, String sort, int userId) {
        return NewsScrapRequest.builder()
                .stockCode(stockCode)
                .page(page)
                .count(count)
                .sort(sort)
                .userId(userId)
                .build();
    }
}
