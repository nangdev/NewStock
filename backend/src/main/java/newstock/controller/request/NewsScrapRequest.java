package newstock.controller.request;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class NewsScrapRequest {

    private Integer stockId;

    private int page;

    private int count;

    private String sort;

    private int userId;

    public static NewsScrapRequest of(Integer stockId, int page, int count, String sort, int userId) {
        return NewsScrapRequest.builder()
                .stockId(stockId)
                .page(page)
                .count(count)
                .sort(sort)
                .userId(userId)
                .build();
    }
}
