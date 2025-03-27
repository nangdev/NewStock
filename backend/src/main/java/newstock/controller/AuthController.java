package newstock.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.dto.Api;
import newstock.controller.request.LoginRequest;
import newstock.controller.response.LoginResponse;
import newstock.domain.user.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/auth")
@Slf4j
public class AuthController {

    private final UserService userService;

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
}
