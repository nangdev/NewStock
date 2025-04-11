package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.response.NotificationListResponse;
import newstock.domain.notification.scheduler.NewsLetterNotificationScheduler;
import newstock.domain.notification.service.NotificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/notification")
public class NotificationController {

    private final NotificationService notificationService;
    private final NewsLetterNotificationScheduler newsLetterNotificationScheduler;

    @GetMapping
    public ResponseEntity<Api<NotificationListResponse>> getUserNotifications(@AuthenticationPrincipal Integer userId) {
        return ResponseEntity.ok(Api.ok(notificationService.getUserNotifications(userId)));
    }

    @PutMapping("/{userNotificationId}")
    public ResponseEntity<Api<Void>> updateUserNotification(@PathVariable Integer userNotificationId) {
        notificationService.updateUserNotifications(userNotificationId);
        return ResponseEntity.ok(Api.ok());
    }

    @DeleteMapping("/{userNotificationId}")
    public ResponseEntity<Api<Void>> deleteUserNotification(@PathVariable Integer userNotificationId) {
        notificationService.deleteUserNotifications(userNotificationId);
        return ResponseEntity.ok(Api.ok());
    }

    @GetMapping("/newsletter/test")
    public ResponseEntity<Api<Void>> testNewsletter() {
        newsLetterNotificationScheduler.sendNewsLetterNotification();
        return ResponseEntity.ok(Api.ok());
    }
}
