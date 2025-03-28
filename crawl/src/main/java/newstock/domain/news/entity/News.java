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
    private Integer newsId;

    private Integer stockId;

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

    private int score;

    public static News of(NewsItem newsItem) {
        return News.builder()
                .stockId(newsItem.getStockId())
                .title(newsItem.getTitle())
                .description(newsItem.getDescription())
                .content(newsItem.getContent())
                .newsImage(newsItem.getNewsImage())
                .url(newsItem.getUrl())
                .press(newsItem.getPress())
                .pressLogo(newsItem.getPressLogo())
                .publishedDate(newsItem.getPublishedDate())
                .newsSummary(newsItem.getNewsSummary())
                .score(newsItem.getScore())
                .build();
    }

}
