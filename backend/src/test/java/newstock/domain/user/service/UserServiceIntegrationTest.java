package newstock.domain.user.service;

import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.LoginRequest;
import newstock.controller.request.UserRequest;
import newstock.domain.user.repository.UserRepository;
import newstock.exception.type.ValidationException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.assertj.core.api.Assertions.*;

@Slf4j
@SpringBootTest
@Transactional
class UserServiceIntegrationTest {

    @Autowired
    private UserService userService;

    @Autowired
    private AuthService authService;

    @Autowired
    private UserRepository userRepository;

    private UserRequest testUser;

    @BeforeEach
    void setUp() {
        testUser = UserRequest.builder()
                .email("test@example.com")
                .password("test1234!")
                .nickname("테스트유저")
                .build();
    }

    @Test
    void 회원가입_로그인_탈퇴_재가입_테스트() {
        userService.addUser(testUser);
        assertThat(userRepository.existsByEmailAndIsActivatedTrue(testUser.getEmail())).isTrue();
        log.info("✅ 회원가입 완료: email: {}", testUser.getEmail());

        LoginRequest loginRequest = new LoginRequest(testUser.getEmail(), testUser.getPassword(), null);
        var loginResponse = authService.login(loginRequest);
        assertThat(loginResponse.getAccessToken()).isNotNull();
        log.info("✅ 로그인 완료: accessToken={}", loginResponse.getAccessToken());

        Integer userId = userRepository.findByEmailAndIsActivatedTrue(testUser.getEmail()).get().getUserId();
        userService.deleteUser(userId, loginResponse.getAccessToken());
        log.info("✅ 회원 탈퇴 완료: userId={}", userId);

        assertThatThrownBy(() -> authService.login(loginRequest))
                .isInstanceOf(ValidationException.class);
        log.info("✅ 탈퇴 유저 로그인 시도 → 예외 발생 확인");

        userService.addUser(testUser);
        assertThat(userRepository.existsByEmailAndIsActivatedTrue(testUser.getEmail())).isTrue();
        log.info("✅ 재가입 완료: {}", testUser.getEmail());

        var reLoginResponse = authService.login(loginRequest);
        assertThat(reLoginResponse.getAccessToken()).isNotNull();
        log.info("✅ 재로그인 완료: accessToken={}", reLoginResponse.getAccessToken());
    }
}

