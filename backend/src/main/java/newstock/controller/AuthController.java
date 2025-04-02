package newstock.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import newstock.common.dto.Api;
import newstock.controller.request.KakaoLoginRequest;
import newstock.controller.request.LoginRequest;
import newstock.controller.request.RefreshRequest;
import newstock.controller.response.LoginResponse;
import newstock.domain.user.service.AuthService;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;

import java.net.URI;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/auth")
@Slf4j
public class AuthController {

    private final AuthService authService;

    /**
     * 로그인 API
     * @param loginRequest 로그인 요청 DTO
     * @return access token, refresh token
     */
    @PostMapping("/login")
    public ResponseEntity<Api<LoginResponse>> login(@RequestBody LoginRequest loginRequest) {
        log.info("로그인 요청: {}", loginRequest.getEmail());
        LoginResponse loginResponse = authService.login(loginRequest);
        return ResponseEntity.ok(Api.ok(loginResponse));
    }

    /**
     * 로그아웃 API
     */
    @PostMapping("/logout")
    public ResponseEntity<Api<Void>> logout(
            @AuthenticationPrincipal Integer userId,
            @RequestHeader("Authorization") String bearerToken) {

        log.info("로그아웃 요청 - userId: {}", userId);

        String token = bearerToken.replace("Bearer ", "");
        authService.logout(userId, token);

        return ResponseEntity.ok(Api.ok());
    }

    /**
     * JWT 토큰 재발급 API (자동 로그인)
     */
    @PostMapping("/refresh")
    public ResponseEntity<Api<LoginResponse>> reissueToken(
            @RequestHeader("Authorization") String bearerToken,
            @RequestBody RefreshRequest request) {

        String refreshToken = bearerToken.replace("Bearer ", "");
        LoginResponse response = authService.reissueToken(refreshToken, request.getFcmToken());

        return ResponseEntity.ok(Api.ok(response));
    }

    /**
     * 카카오 로그인
     */
    @PostMapping("/oauth/kakao/login")
    public ResponseEntity<Api<LoginResponse>> kakaoLogin(@RequestBody KakaoLoginRequest request) {
        LoginResponse response = authService.loginWithKakao(request.getCode(), request.getFcmToken());

        return ResponseEntity.ok(Api.ok(response));
    }
}
