package newstock.controller;

import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.UserRequest;
import newstock.controller.response.UserResponse;
import newstock.domain.user.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/users")
@Slf4j
public class UserController {

    private final UserService userService;

    /**
     * 회원가입 API
     *
     * @param userRequest 회원가입 요청 DTO
     * @return 생성된 사용자 정보
     */
    @PostMapping("/")
    public ResponseEntity<UserResponse> signupUser(@RequestBody UserRequest userRequest) {
        log.info("회원가입 요청: {}", userRequest.getEmail());

        try {
            UserResponse userResponse = userService.signupUser(userRequest);
            return ResponseEntity.ok(userResponse);
        } catch (IllegalArgumentException e) {
            log.warn("회원가입 실패 - 중복 이메일: {}", userRequest.getEmail());
            return ResponseEntity.badRequest().body(null);
        }
    }
}
