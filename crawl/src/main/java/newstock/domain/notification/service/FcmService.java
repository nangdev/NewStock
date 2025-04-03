package newstock.domain.notification.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.auth.oauth2.GoogleCredentials;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NotificationNewsDto;
import newstock.domain.notification.dto.UserDto;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@Slf4j
@RequiredArgsConstructor
public class FcmService {
    
    private final ObjectMapper objectMapper;
    private final RestTemplate restTemplate;
    
    @Value("${fcm.project-id}")
    private String PROJECT_ID;
    
    @Value("${fcm.service-account-file}")
    private String SERVICE_ACCOUNT_FILE;
    
    private String getAccessToken() throws IOException {
        GoogleCredentials googleCredentials = GoogleCredentials
                .fromStream(new ClassPathResource(SERVICE_ACCOUNT_FILE).getInputStream())
                .createScoped(List.of("https://www.googleapis.com/auth/firebase.messaging"));
        
        googleCredentials.refreshIfExpired();
        return googleCredentials.getAccessToken().getTokenValue();
    }

    public void distributeNotification(List<UserDto> userDtos, NotificationNewsDto notificationNewsDto) {
        try {
            List<String> validTokens = userDtos.stream()
                    .map(UserDto::getFcmToken)
                    .toList();
            
            String url = String.format("https://fcm.googleapis.com/v1/projects/%s/messages:send", PROJECT_ID);
            
            // 액세스 토큰 가져오기
            String accessToken = getAccessToken();
            
            // HTTP 헤더 설정
            HttpHeaders headers = new HttpHeaders();
            headers.set("Authorization", "Bearer " + accessToken);
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            int successCount = 0;
            int failureCount = 0;
            
            // 각 토큰별로 메시지 전송 (비효율적이지만 HTTP v1 API에서는 이 방식이 필요)
            for (String token : validTokens) {
                try {
                    Map<String, Object> message = new HashMap<>();
                    
                    Map<String, Object> notification = new HashMap<>();
                    notification.put("title", notificationNewsDto.getTitle());
                    notification.put("body", notificationNewsDto.getDescription());
                    
                    Map<String, String> data = new HashMap<>();
                    data.put("type", "NOTICE");
                    data.put("newsId", Integer.toString(notificationNewsDto.getNewsId()));
                    data.put("click_action", "MOVE_TO_NEWS");
                    
                    Map<String, Object> fcmMessage = new HashMap<>();
                    fcmMessage.put("token", token);
                    fcmMessage.put("notification", notification);
                    fcmMessage.put("data", data);
                    
                    message.put("message", fcmMessage);
                    
                    // HTTP 요청 생성 및 전송
                    HttpEntity<String> entity = new HttpEntity<>(
                            objectMapper.writeValueAsString(message),
                            headers
                    );
                    
                    ResponseEntity<String> response = restTemplate.exchange(
                            url,
                            HttpMethod.POST,
                            entity,
                            String.class
                    );

                    if (response.getStatusCode() == HttpStatus.OK) {
                        successCount++;
                    } else {
                        failureCount++;
                        log.error("토큰 {} 전송 실패: {}", token, response.getBody());
                    }
                } catch (Exception e) {
                    failureCount++;
                    log.error("토큰 {} 전송 중 오류: {}", token, e.getMessage());
                }
            }
            
            log.info("FCM 전송 결과 - 성공: {}, 실패: {}", successCount, failureCount);
            
        } catch (Exception e) {
            log.error("FCM 전송 중 오류 발생: {}", e.getMessage());
        }
    }
}
