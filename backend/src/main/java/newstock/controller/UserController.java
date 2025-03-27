package newstock.controller;

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
    public ResponseEntity<Api<Void>> addUser(@RequestBody UserRequest userRequest) {
        userService.addUser(userRequest);

        return ResponseEntity.ok(Api.ok());
    }

    /**
     * 이메일 중복 체크 API
     */
    @GetMapping("/check-email")
    public ResponseEntity<Api<EmailCheckResponse>> existsByEmail(@RequestParam String email) {
        boolean exists = userService.existsByEmail(email);

        return ResponseEntity.ok(Api.ok(new EmailCheckResponse(exists)));
    }

    @PutMapping("")
    public Api<Void> updateUserRole(@AuthenticationPrincipal CustomUserDetails userDetails) {
        userService.updateUserRole(userDetails.getUser());

        return Api.ok();
    }

    @GetMapping("")
    public Api<UserResponse> getUserInfo(@AuthenticationPrincipal CustomUserDetails userDetails) {
        User user = userDetails.getUser();
        UserResponse userResponse = UserResponse.of(user);

        return Api.ok(userResponse);
    }
}

