package newstock.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.dto.Api;
import newstock.controller.request.EmailRequest;
import newstock.controller.request.EmailVerifyRequest;
import newstock.controller.request.NicknameUpdateRequest;
import newstock.controller.request.UserRequest;
import newstock.controller.response.EmailCheckResponse;
import newstock.controller.response.UserResponse;
import newstock.domain.user.service.EmailSenderService;
import newstock.domain.user.service.EmailVerificationService;
import newstock.domain.user.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/users")
@Slf4j
public class UserController {

    private final UserService userService;
    private final EmailVerificationService emailVerificationService;
    private final EmailSenderService emailSenderService;

    /**
     * 회원가입 API
     *
     * @param userRequest 회원가입 요청 DTO
     * @return 생성된 사용자 정보
     */
    @PostMapping("")
    public ResponseEntity<Api<Void>> addUser(@Valid @RequestBody UserRequest userRequest) {
        userService.addUser(userRequest);

        return ResponseEntity.ok(Api.ok());
    }

    /**
     * 유저 정보 조회 API
     */
    @GetMapping("")
    public ResponseEntity<Api<UserResponse>> getUserInfo(@AuthenticationPrincipal Integer userId) {

        return ResponseEntity.ok(Api.ok(userService.getUserInfo(userId)));
    }

    /**
     * 이메일 중복 체크 API
     */
    @GetMapping("/check-email")
    public ResponseEntity<Api<EmailCheckResponse>> existsByEmail(@RequestParam String email) {
        boolean exists = userService.existsByEmail(email);

        return ResponseEntity.ok(Api.ok(new EmailCheckResponse(exists)));
    }

    /**
     * 최초 로그인 시 유저 권한 변경 API
     */
    @PutMapping("/new")
    public ResponseEntity<Api<Void>> updateUserRole(@AuthenticationPrincipal Integer userId) {
        userService.updateUserRole(userId);

        return ResponseEntity.ok(Api.ok());
    }

    /**
     * 이메일 인증 요청
     */
    @PostMapping("/send-email")
    public ResponseEntity<Api<Void>> sendEmailCode(@Valid @RequestBody EmailRequest request) {
        log.info("이메일 인증 요청 - email: {}", request.getEmail());
        String code = emailVerificationService.generateAndSaveAuthCode(request.getEmail());
        emailSenderService.sendEmailCode(request.getEmail(), code);

        return ResponseEntity.ok(Api.ok());
    }

    /**
     * 이메일 인증 확인
     */
    @PostMapping("/verify-email")
    public ResponseEntity<Api<Void>> verifyEmailCode(@Valid @RequestBody EmailVerifyRequest request) {
        log.info("이메일 인증 확인 - email: {}, code: {}", request.getEmail(), request.getAuthCode());
        emailVerificationService.verifyAuthCode(request.getEmail(), request.getAuthCode());

        return ResponseEntity.ok(Api.ok());
    }

    /**
     * 닉네임 변경
     */
    @PutMapping("/nickname")
    public ResponseEntity<Api<UserResponse>> updateNickname(
            @AuthenticationPrincipal Integer userId,
            @Valid @RequestBody NicknameUpdateRequest request) {
        log.info("닉네임 변경 요청 - userId: {}", userId);
        UserResponse response = userService.updateNickname(userId, request.getNickname());

        return ResponseEntity.ok(Api.ok(response));
    }

    /**
     * 회원 탈퇴
     */
    @DeleteMapping("")
    public ResponseEntity<Api<Void>> deleteUser(
            @AuthenticationPrincipal Integer userId,
            @RequestHeader("Authorization") String bearerToken) {

        log.info("탈퇴 요청 - userId: {}", userId);

        String accessToken = bearerToken.replace("Bearer ", "");
        userService.deleteUser(userId, accessToken);

        return ResponseEntity.ok(Api.ok());
    }
}