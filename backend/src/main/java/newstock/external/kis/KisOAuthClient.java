package newstock.external.kis;

import lombok.RequiredArgsConstructor;
import newstock.external.kis.request.KisAccessTokenRequest;
import newstock.external.kis.response.KisAccessTokenResponse;
import newstock.external.kis.request.KisWebSocketKeyRequest;
import newstock.external.kis.response.KisWebSocketKeyResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
public class KisOAuthClient {
    @Value("${kis.app-key}")
    private String appKey;
    @Value("${kis.secret-key}")
    private String secretKey;

    private final WebClient webClient;

    private static final String BASE_URL = "https://openapi.koreainvestment.com";
    private static final int PORT = 9443;   // 실전투자계좌포트
    private static final String GRANT_TYPE = "client_credentials";

    public KisWebSocketKeyResponse getWebSocketKey() {
        String endPoint = "/oauth2/Approval";

        KisWebSocketKeyRequest requestDto = new KisWebSocketKeyRequest();
        requestDto.setGrantType(GRANT_TYPE);
        requestDto.setAppkey(appKey);
        requestDto.setSecretkey(secretKey);

        return webClient.post()
                .uri(BASE_URL+":"+PORT+endPoint)
                .bodyValue(requestDto)
                .retrieve()
                .bodyToMono(KisWebSocketKeyResponse.class)
                .block();
    }

    public KisAccessTokenResponse getAccessToken() {
        String endPoint = "/oauth2/tokenP";

        KisAccessTokenRequest requestDto = new KisAccessTokenRequest();
        requestDto.setGrantType(GRANT_TYPE);
        requestDto.setAppkey(appKey);
        requestDto.setAppsecret(secretKey);

        return webClient.post()
                .uri(BASE_URL+":"+PORT+endPoint)
                .bodyValue(requestDto)
                .retrieve()
                .bodyToMono(KisAccessTokenResponse.class)
                .block();
    }
}
