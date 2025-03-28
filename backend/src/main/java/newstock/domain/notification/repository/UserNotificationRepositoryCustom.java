package newstock.domain.notification.repository;

import newstock.domain.notification.dto.UserNotificationDto;

import java.util.List;

public interface UserNotificationRepositoryCustom {

    List<UserNotificationDto>  findUserNotificationsWithDetails(Integer userId);

}
