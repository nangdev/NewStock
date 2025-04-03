package newstock.domain.notification.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.notification.dto.NotificationDto;
import newstock.domain.notification.dto.NotificationResultDto;
import newstock.domain.notification.entity.Notification;
import newstock.domain.notification.entity.UserNotification;
import newstock.domain.notification.repository.NotificationRepository;
import newstock.domain.notification.repository.UserNotificationRepository;
import newstock.domain.stock.repository.UserStockRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {

    private final NotificationRepository notificationRepository;
    private final UserStockRepository userStockRepository;
    private final UserNotificationRepository userNotificationRepository;
    private final FcmService fcmService;

    @Transactional
    public void addNotifications(NotificationDto notificationDto) {
        Integer stockId = notificationDto.getStockId();
        Integer newsId = notificationDto.getNewsId();

        NotificationResultDto result = userStockRepository.findUsersAndNewsById(stockId, newsId);

        if (result.getUserDtos().isEmpty()) {
            log.info("알림을 받을 사용자가 없습니다. stockId: {}", stockId);
            return;
        }

        Notification notification = notificationRepository.save(new Notification(stockId, newsId));

        List<UserNotification> userNotifications = result.getUserDtos().stream()
                .map(userDto -> UserNotification.of(notification.getNotificationId(), userDto.getUserId()))
                .collect(Collectors.toList());

        userNotificationRepository.saveAll(userNotifications);  // 벌크 저장

        fcmService.distributeNotification(result.getUserDtos(), result.getNewsDto());
    }

}
