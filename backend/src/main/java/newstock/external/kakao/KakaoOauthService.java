package newstock.external.kakao;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.exception.type.ValidationException;
import newstock.exception.ExceptionCode;
import newstock.external.kakao.dto.KakaoTokenResponse;
import newstock.external.kakao.dto.KakaoUserInfo;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;

@Slf4j
@Service
@RequiredArgsConstructor
public class KakaoOauthService {

    @Value("${kakao.client-id}")
    private String clientId;

    @Value("${kakao.redirect-uri}")
    private String redirectUri;

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${kakao.redirect-uri}")
    private String kakaoRedirectUri;

    // 인가 코드로 카카오 토큰 요청
    public KakaoTokenResponse getToken(String code) {

        String tokenUrl = "https://kauth.kakao.com/oauth/token";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        params.add("grant_type", "authorization_code");
        params.add("client_id", clientId);
        params.add("redirect_uri", redirectUri);
        params.add("code", code);

        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(params, headers);

        try {
            ResponseEntity<KakaoTokenResponse> response = restTemplate.exchange(
                    tokenUrl,
                    HttpMethod.POST,
                    request,
                    KakaoTokenResponse.class
            );
            return response.getBody();
        } catch (HttpClientErrorException e) {
            log.error("카카오 토큰 요청 실패: {}", e.getResponseBodyAsString());
            throw new ValidationException(ExceptionCode.KAKAO_TOKEN_ERROR);
        }
    }

    /**
     * 카카오 토큰으로 카카오 사용자 정보 조회
     */
    public KakaoUserInfo getUserInfo(String accessToken) {

        String userInfoUrl = "https://kapi.kakao.com/v2/user/me";

        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(accessToken);

        HttpEntity<Void> request = new HttpEntity<>(headers);

        try {
            ResponseEntity<KakaoUserInfo> response = restTemplate.exchange(
                    userInfoUrl, HttpMethod.GET, request, KakaoUserInfo.class
            );
            return response.getBody();
        } catch (HttpClientErrorException e) {
            log.error("카카오 사용자 정보 조회 실패: {}", e.getResponseBodyAsString());
            throw new ValidationException(ExceptionCode.KAKAO_USERINFO_ERROR);
        }
    }
}
