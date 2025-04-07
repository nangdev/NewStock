package newstock.domain.user.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.redis.RedisUtil;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.stereotype.Service;

import java.util.Random;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmailVerificationService {

    private final RedisUtil redisUtil;

    private static final String EMAIL_AUTH_PREFIX = "email:auth:";
    private static final long EXPIRATION_TIME = 60 * 5L; // 5분 TTL

    /**
     * 인증 번호 생성 (6자리 숫자)
     */
    private String generateAuthCode() {
        Random random = new Random();
        return String.format("%06d", random.nextInt(1000000));
    }

    /**
     * 이메일 인증 번호 생성 및 Redis에 저장
     */
    public String generateAndSaveAuthCode(String email) {

        String authCode = generateAuthCode();
        redisUtil.set(EMAIL_AUTH_PREFIX + email, authCode, EXPIRATION_TIME);
        log.info("이메일 인증 번호 생성 - email: {}, code: {}", email, authCode);

        return authCode;
    }

    /**
     * 인증 번호 검증 및 이메일 인증 처리
     */
    public void verifyAuthCode(String email, String inputCode) {
        String key = EMAIL_AUTH_PREFIX + email;
        String savedCode = redisUtil.get(key, String.class);

        if (savedCode == null) {
            throw new ValidationException(ExceptionCode.EMAIL_AUTH_EXPIRED);
        }

        if (!savedCode.equals(inputCode)) {
            throw new ValidationException(ExceptionCode.EMAIL_AUTH_INVALID);
        }

        redisUtil.delete(key);
        log.info("이메일 인증 성공 - email: {}", email);

        redisUtil.set("email:verified:" + email, true, 60 * 10); // 이메일 인증 10분간 유효
    }
}