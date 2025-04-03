package newstock.domain.notification.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NotificationNewsDto;
import newstock.domain.notification.dto.UserDto;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@Slf4j
@RequiredArgsConstructor
public class FcmService {

    @Value("${fcm.server.key}")
    private String FCM_SERVER_KEY;

    @Value("${fcm.api.url}")
    private String FCM_API_URL;

    private final ObjectMapper objectMapper;
    private final RestTemplate restTemplate;

    public void distributeNotification(List<UserDto> userDtos, NotificationNewsDto notificationNewsDto) {
        try {
            List<String> validTokens = userDtos.stream()
                    .map(UserDto::getFcmToken)
                    .collect(Collectors.toList());

            // FCM 메시지 구성
            Map<String, Object> notification = new HashMap<>();
            notification.put("title", notificationNewsDto.getTitle());
            notification.put("body", notificationNewsDto.getDescription());

            Map<String, Object> message = new HashMap<>();
            message.put("registration_ids", validTokens);  // 여러 토큰을 한번에 전송
            message.put("notification", notification);

            // 추가 데이터
            Map<String, String> data = new HashMap<>();
            data.put("type", "NOTICE");
            data.put("newsId", Integer.toString(notificationNewsDto.getNewsId()));
            data.put("click_action", "MOVE_TO_NEWS");
            message.put("data", data);

            // HTTP 헤더 설정
            HttpHeaders headers = new HttpHeaders();
            headers.set("Authorization", "key=" + FCM_SERVER_KEY);
            headers.setContentType(MediaType.APPLICATION_JSON);

            // HTTP 요청 생성 및 전송
            HttpEntity<String> entity = new HttpEntity<>(
                    objectMapper.writeValueAsString(message),
                    headers
            );

            ResponseEntity<String> response = restTemplate.exchange(
                    FCM_API_URL,
                    HttpMethod.POST,
                    entity,
                    String.class
            );

            // 응답 처리
            if (response.getStatusCode() == HttpStatus.OK) {
                JsonNode responseBody = objectMapper.readTree(response.getBody());
                int success = responseBody.get("success").asInt();
                int failure = responseBody.get("failure").asInt();

                log.info("FCM 멀티캐스트 전송 결과 - 성공: {}, 실패: {}", success, failure);

                // 실패한 토큰 처리
                if (failure > 0) {
                    JsonNode results = responseBody.get("results");
                    for (int i = 0; i < results.size(); i++) {
                        JsonNode result = results.get(i);
                        if (result.has("error")) {
                            String errorToken = validTokens.get(i);
                            String errorCode = result.get("error").asText();
                            log.error("토큰 {} 전송 실패: {}", errorToken, errorCode);
                        }
                    }
                }
            } else {
                log.error("FCM 멀티캐스트 전송 실패: {}", response.getBody());
            }

        } catch (Exception e) {
            log.error("FCM 멀티캐스트 전송 중 오류 발생: {}", e.getMessage());
        }
    }
}
