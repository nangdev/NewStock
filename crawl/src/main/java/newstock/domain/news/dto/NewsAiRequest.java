package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class NewsAiRequest {

    private String stockName;

    private List<NewsItem> newsItemList;

    public static NewsAiRequest of(String stockName, List<NewsItem> newsItemList) {
        return NewsAiRequest.builder()
                .stockName(stockName)
                .newsItemList(newsItemList)
                .build();
    }
}
