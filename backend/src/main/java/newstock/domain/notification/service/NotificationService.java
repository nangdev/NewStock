package newstock.domain.notification.service;

import lombok.RequiredArgsConstructor;
import newstock.controller.response.NotificationListResponse;
import newstock.domain.news.entity.News;
import newstock.domain.news.repository.NewsRepository;
import newstock.domain.notification.dto.NewsNotificationDto;
import newstock.domain.notification.dto.StockNotificationDto;
import newstock.domain.notification.dto.UserNotificationDto;
import newstock.domain.notification.entity.Notification;
import newstock.domain.notification.entity.UserNotification;
import newstock.domain.notification.repository.NotificationRepository;
import newstock.domain.notification.repository.UserNotificationRepository;
import newstock.domain.stock.entity.Stock;
import newstock.domain.stock.repository.StockRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class NotificationService {

    private final UserNotificationRepository userNotificationRepository;

    private final NotificationRepository notificationRepository;

    private final StockRepository stockRepository;

    private final NewsRepository newsRepository;

    public NotificationListResponse getUserNotifications(Integer userId) {

        List<UserNotification> userNotifications = userNotificationRepository.findAllByUserId(userId);

        List<UserNotificationDto> notificationList = new ArrayList<>();

        for (UserNotification un : userNotifications) {
            Notification notification = notificationRepository.findById(un.getNotificationId())
                    .orElseThrow(() -> new RuntimeException("Notification not found for id: " + un.getNotificationId()));

            StockNotificationDto stockNotificationDto = StockNotificationDto.of(stockRepository.findByStockId(notification.getStockId()));
            NewsNotificationDto newsNotificationDto = NewsNotificationDto.of(newsRepository.findByNewsId(notification.getNewsId()));

            UserNotificationDto dto = UserNotificationDto.builder()
                    .unId(un.getUnId())
                    .stockInfo(stockNotificationDto)
                    .newsInfo(newsNotificationDto)
                    .isRead(un.getIsRead() != null && un.getIsRead() == 1)
                    .build();

            notificationList.add(dto);
        }

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
