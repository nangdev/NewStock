package newstock.domain.news.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class NewsItem {

    private int id;

    private String stockCode;

    private String title;

    private String description;

    private String content;

    private String newsImage;

    private String url;

    private String press;

    private String pressLogo;

    private String publishedDate;

    private String newsSummary;

    private String score;

    private String keyword;

}
