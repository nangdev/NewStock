package newstock.external.kis;

import lombok.RequiredArgsConstructor;
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

    public KisWebSocketKeyResponse getWebSocketKey() {
        String endPoint = "/oauth2/Approval";
        String grantType = "client_credentials";

        KisWebSocketKeyRequest requestDto = new KisWebSocketKeyRequest();
        requestDto.setGrantType(grantType);
        requestDto.setAppkey(appKey);
        requestDto.setSecretkey(secretKey);

        return webClient.post()
                .uri(BASE_URL+":"+PORT+endPoint)
                .bodyValue(requestDto)
                .retrieve()
                .bodyToMono(KisWebSocketKeyResponse.class)
                .block();
    }


}
