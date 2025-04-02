package newstock.domain.user.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.redis.RedisUtil;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.stereotype.Service;

import java.util.Random;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmailVerificationService {

    private final RedisUtil redisUtil;
    private final UserRepository userRepository;

    private static final String EMAIL_AUTH_PREFIX = "email:auth:";
    private static final long EXPIRATION_TIME = 60 * 5L; // 5분 TTL

    /**
     * 인증번호 생성 (6자리 숫자)
     */
    private String generateAuthCode() {
        Random random = new Random();
        return String.valueOf(100000 + random.nextInt(900000));// 100000~999999
    }

    /**
     * 이메일 인증 코드 생성 및 Redis에 저장
     */
    public String generateAndSaveAuthCode(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        if (user.isEmailVerified()) {
            throw new ValidationException(ExceptionCode.EMAIL_ALREADY_VERIFIED);
        }

        String authCode = generateAuthCode();
        redisUtil.set(EMAIL_AUTH_PREFIX + email, authCode, EXPIRATION_TIME);
        log.info("이메일 인증 코드 생성 - email: {}, code: {}", email, authCode);

        return authCode;
    }

    /**
     * 인증번호 검증 및 이메일 인증 처리
     */
    public void verifyAuthCode(String email, String inputCode) {
        String key = EMAIL_AUTH_PREFIX + email;
        String savedCode = redisUtil.get(key);

        if (savedCode == null) {
            throw new ValidationException(ExceptionCode.EMAIL_AUTH_EXPIRED);
        }

        if (!savedCode.equals(inputCode)) {
            throw new ValidationException(ExceptionCode.EMAIL_AUTH_INVALID);
        }

        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        if (user.isEmailVerified()) {
            throw new ValidationException(ExceptionCode.EMAIL_ALREADY_VERIFIED);
        }

        user.setEmailVerified(true);
        userRepository.save(user);

        redisUtil.delete(key);
        log.info("이메일 인증 성공 - email: {}", email);
    }
}
