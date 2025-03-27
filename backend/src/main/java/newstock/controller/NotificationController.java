package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.response.NotificationListResponse;
import newstock.domain.notification.service.NotificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/notification")
public class NotificationController {

    private final NotificationService notificationService;

    @GetMapping
    public ResponseEntity<Api<NotificationListResponse>> getUserNotifications(@AuthenticationPrincipal Integer userId) {
        return ResponseEntity.ok(Api.ok(notificationService.getUserNotifications(userId)));
    }

}
