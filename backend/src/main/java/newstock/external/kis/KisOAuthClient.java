package newstock.external.kis;

import lombok.RequiredArgsConstructor;
import newstock.external.kis.dto.KisStockInfoDto;
import newstock.external.kis.request.KisAccessTokenRequest;
import newstock.external.kis.request.KisWebSocketKeyRequest;
import newstock.external.kis.response.KisAccessTokenResponse;
import newstock.external.kis.response.KisWebSocketKeyResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

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
                .uri(BASE_URL + ":" + PORT + endPoint)
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
                .uri(BASE_URL + ":" + PORT + endPoint)
                .bodyValue(requestDto)
                .retrieve()
                .bodyToMono(KisAccessTokenResponse.class)
                .block();
    }

    public KisStockInfoDto getStockInfo(String stockCode, String accessToken) {
        String endPoint = "/uapi/domestic-stock/v1/quotations/inquire-price";
        String queryParam = "?FID_COND_MRKT_DIV_CODE=J&FID_INPUT_ISCD=" + stockCode;

        try {
            // API 응답을 Map으로 받아서 필요한 값만 추출
            Map<String, Object> response = webClient.get()
                    .uri(BASE_URL + ":" + PORT + endPoint + queryParam)
                    .header("content-type", "application/json; charset=utf-8")
                    .header("authorization", "Bearer " + accessToken)
                    .header("appkey", appKey)
                    .header("appsecret", secretKey)
                    .header("tr_id", "FHKST01010100")  // 주식현재가 시세 API의 tr_id
                    .header("custtype", "P")           // 개인
                    .retrieve()
                    .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {
                    })
                    .block();

            Map<String, Object> output = (Map<String, Object>) response.get("output");

            return KisStockInfoDto.builder()
                    .closingPrice((String) output.get("stck_prpr"))    // 주식 현재가
                    .ctpdPrice((String) output.get("prdy_vrss"))       // 전일 대비
                    .rcPdcp((String) output.get("prdy_ctrt"))          // 전일 대비율
                    .totalPrice((String) output.get("hts_avls"))       // HTS 시가총액
                    .build();
        } catch (Exception e) {
            // 예외 처리
            System.out.println("주식 [" + stockCode + "] 정보 조회 중 오류 발생: " + e.getMessage());
            return null;
        }
    }

}
