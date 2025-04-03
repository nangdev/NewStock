package newstock.domain.notification.dto;

import lombok.Builder;
import lombok.Data;
import newstock.domain.news.dto.NewsItem;

@Data
@Builder
public class NotificationDto {

    private Integer newsId;

    private Integer stockId;

    public static NotificationDto of(NewsItem newsItem) {
        return NotificationDto.builder()
                .newsId(newsItem.getId())
                .stockId(newsItem.getStockId())
                .build();
    }

}
