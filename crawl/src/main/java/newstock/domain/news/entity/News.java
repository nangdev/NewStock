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
    private int id;

    private String stockCode;

    private String title;

    private String description;

    @Lob
    private String content;

    private String newsImage;

    private String url;

    private String press;

    private String pressLogo;

    private String publishedDate;

    @Lob
    private String newsSummary;

    private String score;

    private String keyword;

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
