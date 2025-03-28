package newstock.kafka.request;

import lombok.Builder;
import lombok.Data;
import newstock.domain.news.dto.NewsItem;

import java.util.List;

@Data
@Builder
public class NewsDbRequest {

    private String stockName;

    private List<NewsItem> filteredNewsItems;

    public static NewsDbRequest of(String stockName,List<NewsItem> newsItems) {
        return NewsDbRequest.builder()
                .stockName(stockName)
                .filteredNewsItems(newsItems)
                .build();
    }

}
