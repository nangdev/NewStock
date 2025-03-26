package newstock.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.UserRequest;
import newstock.controller.response.UserResponse;
import newstock.domain.user.service.UserService;
import newstock.exception.type.DuplicateEmailException;
import org.springframework.http.ResponseEntity;
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
    public ResponseEntity<UserResponse> signupUser(@RequestBody UserRequest userRequest) {
        log.info("회원가입 요청: {}", userRequest.getEmail());
        // 이메일 중복
        if (userService.existsByEmail(userRequest.getEmail())) {
            throw new DuplicateEmailException();
        }

        UserResponse userResponse = userService.signupUser(userRequest);
        return ResponseEntity.ok(userResponse);
    }

    /**
     * 이메일 중복 체크 API
     *
     * @param email 확인할 이메일
     * @return 중복 여부 (true = 중복, false = 사용 가능)
     */
    @GetMapping("/check-email")
    public ResponseEntity<Boolean> existsByEmail(@RequestParam String email) {
        log.info("이메일 중복 체크 요청: {}", email);
        boolean exists = userService.existsByEmail(email);
        if (exists) {
            log.warn("이메일 중복됨: {}", email);
        } else {
            log.info("사용 가능한 이메일입니다: {}", email);
        }

        return ResponseEntity.ok(exists);
    }
}

