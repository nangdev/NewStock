package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.dto.StockNewsDto;

import java.util.List;

@Getter
@Builder
public class StockNewsResponse {

    private int totalPage;

    private List<StockNewsDto> newsList;

    public static StockNewsResponse of(int totalPage,List<StockNewsDto> stockNewsDtoList) {
        return StockNewsResponse.builder()
                .totalPage(totalPage)
                .newsList(stockNewsDtoList)
                .build();
    }
}
