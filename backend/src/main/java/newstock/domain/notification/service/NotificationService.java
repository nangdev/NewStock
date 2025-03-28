package newstock.domain.notification.service;

import lombok.RequiredArgsConstructor;
import newstock.controller.response.NotificationListResponse;
import newstock.domain.notification.entity.UserNotification;
import newstock.domain.notification.repository.NotificationRepository;
import newstock.domain.notification.repository.UserNotificationRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class NotificationService {

    private final NotificationRepository notificationRepository;
    private final UserNotificationRepository userNotificationRepository;

    public NotificationListResponse getUserNotifications(Integer userId) {
        List<UserNotification> unList = userNotificationRepository.findAllByUserId(userId);


        return null;
    }

    @Transactional
    public void updateUserNotifications(Integer userId, Integer notificationId) {

    }

    @Transactional
    public void deleteUserNotifications(Integer userId, Integer notificationId) {

    }

}
