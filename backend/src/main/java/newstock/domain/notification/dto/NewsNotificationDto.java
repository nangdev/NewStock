package newstock.domain.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.news.entity.News;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsNotificationDto {

    private Integer newsId;

    private String title;

    private String publishedDate;

    public static NewsNotificationDto of(News news) {
        return NewsNotificationDto.builder()
                .newsId(news.getNewsId())
                .title(news.getTitle())
                .publishedDate(news.getPublishedDate())
                .build();
    }
}
