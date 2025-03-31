package newstock.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.dto.Api;
import newstock.common.jwt.JwtTokenProvider;
import newstock.common.jwt.TokenBlacklistService;
import newstock.controller.request.LoginRequest;
import newstock.controller.request.RefreshRequest;
import newstock.controller.response.LoginResponse;
import newstock.domain.user.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/auth")
@Slf4j
public class AuthController {

    private final UserService userService;
    private final JwtTokenProvider jwtTokenProvider;
    private final TokenBlacklistService tokenBlacklistService;

    /**
     * 로그인 API
     * @param loginRequest 로그인 요청 DTO
     * @return access token, refresh token
     */
    @PostMapping("/login")
    public ResponseEntity<Api<LoginResponse>> login(@RequestBody LoginRequest loginRequest) {
        log.info("로그인 요청: {}", loginRequest.getEmail());
        LoginResponse loginResponse = userService.login(loginRequest);
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
        userService.logout(userId, token);

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
        LoginResponse response = userService.reissueToken(refreshToken, request.getFcmToken());

        return ResponseEntity.ok(Api.ok(response));
    }
}
