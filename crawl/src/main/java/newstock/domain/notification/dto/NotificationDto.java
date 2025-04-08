package newstock.domain.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.news.dto.NewsItem;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
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
