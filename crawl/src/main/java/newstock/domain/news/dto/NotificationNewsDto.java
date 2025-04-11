package newstock.domain.news.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class NotificationNewsDto {

    private Integer newsId;

    private String title;

    private String description;

}
