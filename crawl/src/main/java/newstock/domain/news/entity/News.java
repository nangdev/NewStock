package newstock.domain.news.entity;

import jakarta.persistence.*;
import lombok.*;
import newstock.domain.news.dto.NewsItem;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name="news")
public class News {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    int id;

    String stockCode;

    String title;

    String description;

    @Lob
    String content;

    String newsImage;

    String url;

    String press;

    String pressLogo;

    String publishedDate;

    @Lob
    String newsSummary;

    String score;

    String keyword;

    public static News of(NewsItem newsItem) {
        return News.builder()
                .stockCode(newsItem.getStockCode())
                .title(newsItem.getTitle())
                .description(newsItem.getDescription())
                .content(newsItem.getContent())
                .newsImage(newsItem.getNewsImage())
                .url(newsItem.getUrl())
                .press(newsItem.getPress())
                .pressLogo(newsItem.getPressLogo())
                .publishedDate(newsItem.getPublishedDate())
                .newsSummary(newsItem.getNewsSummary())
                .score(null)
                .keyword(null)
                .build();
    }

}
