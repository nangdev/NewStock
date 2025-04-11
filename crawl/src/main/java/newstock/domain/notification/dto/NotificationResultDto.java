package newstock.domain.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.news.dto.NotificationNewsDto;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class NotificationResultDto {

    private List<UserDto> userDtos;

    private NotificationNewsDto newsDto;

}