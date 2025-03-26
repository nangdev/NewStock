package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.dto.StockNewsDto;

import java.util.List;

@Getter
@Builder
public class NewsScrapResponse {

    private int totalPage;

    private List<StockNewsDto> newsList;

    public static NewsScrapResponse of(int totalPage,List<StockNewsDto> stockNewsDtoList) {
        return NewsScrapResponse.builder()
                .totalPage(totalPage)
                .newsList(stockNewsDtoList)
                .build();
    }
}