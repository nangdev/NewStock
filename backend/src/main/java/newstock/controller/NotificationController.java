package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.response.NotificationListResponse;
import newstock.domain.notification.service.NotificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/notification")
public class NotificationController {

    private final NotificationService notificationService;

    @GetMapping
    public ResponseEntity<Api<NotificationListResponse>> getUserNotifications(@AuthenticationPrincipal Integer userId) {
        return ResponseEntity.ok(Api.ok(notificationService.getUserNotifications(userId)));
    }

    @PutMapping("/{notificationId}")
    public ResponseEntity<Api<Void>> updateUserNotification(@PathVariable Integer notificationId,
                                                            @AuthenticationPrincipal Integer userId) {
        notificationService.updateUserNotifications(notificationId, userId);
        return ResponseEntity.ok(Api.ok());
    }

    @DeleteMapping("/{notificationId}")
    public ResponseEntity<Api<Void>> deleteUserNotification(@PathVariable Integer notificationId,
                                                            @AuthenticationPrincipal Integer userId) {
        notificationService.deleteUserNotifications(notificationId, userId);
        return ResponseEntity.ok(Api.ok());
    }
}
