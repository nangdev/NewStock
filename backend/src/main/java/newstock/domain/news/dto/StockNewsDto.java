package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;
import newstock.domain.news.entity.News;

@Data
@Builder
public class StockNewsDto {

    private Integer newsId;

    private String title;

    private String description;

    private int score;

    private String publishedDate;

    public static StockNewsDto of(News news) {
        return StockNewsDto.builder()
                .newsId(news.getNewsId())
                .title(news.getTitle())
                .description(news.getDescription())
                .publishedDate(news.getPublishedDate())
                .score(news.getScore())
                .build();
    }
}
