package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;
import newstock.domain.news.entity.News;

@Data
@Builder
public class TopNewsDto {

    private int newsId;

    private String title;

    private String publishedDate;

    private String score;

    public static TopNewsDto of(News news) {
        return TopNewsDto.builder()
                .newsId(news.getNewsId())
                .title(news.getTitle())
                .publishedDate(news.getPublishedDate())
                .score(news.getScore())
                .build();
    }


}
