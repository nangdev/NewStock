package newstock.kafka.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.news.dto.NewsItem;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
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
