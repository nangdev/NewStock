package newstock.domain.notification.service;

import lombok.RequiredArgsConstructor;
import newstock.controller.response.NotificationListResponse;
import newstock.domain.notification.dto.UserNotificationDto;
import newstock.domain.notification.entity.UserNotification;
import newstock.domain.notification.repository.UserNotificationRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class NotificationService {

    private final UserNotificationRepository userNotificationRepository;

    public NotificationListResponse getUserNotifications(Integer userId) {
        List<UserNotificationDto> notificationList = userNotificationRepository.findUserNotificationsWithDetails(userId);

        return NotificationListResponse.builder()
                .notificationList(notificationList)
                .build();
    }

    @Transactional
    public void updateUserNotifications(Integer userNotificationId) {
        UserNotification userNotification = userNotificationRepository.findById(userNotificationId)
                .orElseThrow(() -> new DbException(ExceptionCode.USER_NOTIFICATION_NOT_FOUND));
        userNotification.setIsRead((byte) 1);
    }

    @Transactional
    public void deleteUserNotifications(Integer userNotificationId) {
        userNotificationRepository.deleteById(userNotificationId);
    }

}
