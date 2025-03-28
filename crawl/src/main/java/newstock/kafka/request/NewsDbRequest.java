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
