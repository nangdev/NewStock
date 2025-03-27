package newstock.domain.news.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class NewsItem {

    private Integer id;

    private Integer stockId;

    private String title;

    private String description;

    private String content;

    private String newsImage;

    private String url;

    private String press;

    private String pressLogo;

    private String publishedDate;

    private String newsSummary;

    private int score;

}
