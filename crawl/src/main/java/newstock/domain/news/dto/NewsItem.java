package newstock.domain.news.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class NewsItem {

    int id;

    String stockCode;

    String title;

    String description;

    String content;

    String newsImage;

    String url;

    String press;

    String pressLogo;

    String publishedDate;

    String newsSummary;

    String score;

    String keyword;


}
