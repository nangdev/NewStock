package newstock.domain.notification.service;

import lombok.RequiredArgsConstructor;
import newstock.controller.response.NotificationListResponse;
import newstock.domain.notification.repository.NotificationRepository;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class NotificationService {

    private final NotificationRepository notificationRepository;

    public NotificationListResponse getUserNotifications(Integer userId) {


        return null;
    }

}
