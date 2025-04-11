package newstock.domain.notification.controller;


import lombok.RequiredArgsConstructor;
import newstock.domain.notification.dto.NotificationDto;
import newstock.domain.notification.service.NotificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/noti")
public class NotificationController {

    private final NotificationService notificationService;

    @GetMapping
    public ResponseEntity<Void> sendNotification(@RequestParam("newsId") Integer newsId, @RequestParam("stockId") Integer stockId) {
        notificationService.addNotifications(new NotificationDto(newsId, stockId));
        return ResponseEntity.ok().build();
    }

}
