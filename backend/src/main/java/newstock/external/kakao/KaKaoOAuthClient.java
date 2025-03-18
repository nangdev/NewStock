package newstock.external.kakao;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
public class KaKaoOAuthClient {

    @Value("${kakao.client-id}")
    private String clientId;

    private final WebClient webClient;

    private static final String USER_INFO_URL = "https://kapi.kakao.com/v2/user/me";

    public String getUserToken(String accessToken) {
        return webClient.get()
                .uri(USER_INFO_URL)
                .headers(headers -> headers.setBearerAuth(accessToken))
                .retrieve()
                .bodyToMono(String.class)
                .block();
    }

}
