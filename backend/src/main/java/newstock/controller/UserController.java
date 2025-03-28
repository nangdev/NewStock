package newstock.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.dto.Api;
import newstock.controller.request.UserRequest;
import newstock.controller.response.EmailCheckResponse;
import newstock.controller.response.UserResponse;
import newstock.domain.user.entity.User;
import newstock.domain.user.service.CustomUserDetails;
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

    @PutMapping("/new")
    public ResponseEntity<Api<Void>> updateUserRole(@AuthenticationPrincipal Integer userId) {
        userService.updateUserRole(userId);

        return ResponseEntity.ok(Api.ok());
    }


}

