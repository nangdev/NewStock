package newstock.domain.notification.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class NewsNotificationDto {

    private Integer newsId;

    private String title;

    private String publishedDate;
}
