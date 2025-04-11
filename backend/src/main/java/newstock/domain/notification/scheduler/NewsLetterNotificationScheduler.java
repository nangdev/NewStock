package newstock.domain.notification.scheduler;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.newsletter.repository.NewsletterRepository;
import newstock.domain.notification.service.FcmService;
import newstock.domain.user.repository.UserRepository;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class NewsLetterNotificationScheduler {

    private final FcmService fcmService;
    private final NewsletterRepository newsletterRepository;
    private final UserRepository userRepository;

    @Scheduled(cron = "0 00 18 * * MON-FRI")
    public void sendNewsLetterNotification() {
        log.info("뉴스레터 알림 발송 시작");

        LocalDateTime now = LocalDateTime.now();
        String formattedDate = now.format(DateTimeFormatter.ofPattern("yyMMdd"));

        if(!newsletterRepository.existsByDate(formattedDate)) {
            log.info("{} 자 뉴스레터 없음",formattedDate);
            return;
        }

        List<String> userTokens = userRepository.findUserTokens();

        fcmService.distributeNotification(userTokens, formattedDate);
    }

}
