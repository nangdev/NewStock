package newstock.domain.user.service;

import ch.qos.logback.classic.layout.TTLLLayout;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.redis.RedisUtil;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.stereotype.Service;

import java.util.Random;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmailVerificationService {

    private final RedisUtil redisUtil;
    private final UserRepository userRepository;

    private static final String EMAIL_AUTH_PREFIX = "email:auth:";
    private static final long EXPIRATION_TIME = 60 * 5L; // 5분 TTL
    private static final int CODE_LENGTH = 6;

    /**
     * 인증번호 생성 (6자리 숫자)
     */
    private String generateAuthCode() {
        Random random = new Random();
        return String.valueOf(100000 + random.nextInt(900000));// 100000~999999
    }

    /**
     * 메일 전송
     */
    public void sendEmailCode(String email) {
        String authCode = generateAuthCode();

        MimeMessage message = javaMailSender.createMimeMessage();
        try {
            message.setFrom("team.newstock@gmail.com");
            message.setRecipients(MimeMesage.RecipientType.TO, email);
            message.setSubject("[NewStock] 이메일 인증 코드입니다.");

            String html = "<h3>요청하신 인증번호입니다.</h3>" +
                    "<h1>" + authCode + "</h1>" +
                    "<p>5분 이내로 입력해주세요.</p>";

            message.setText(html, "UTF-8", "html");

            javaMailSender.send(message);

            // Redis에 저장
            redisUtil.set(EMAIL_AUTH_PREFIX + email, authCode, TTL);
            log.info("이메일 전송 및 인증 코드 저장 성공 - email: {}, code: {}", email, authCode);

        } catch (MessagingException e) {
            log.error("이메일 전송 실패 - email: {}", email, e);
            throw new RuntimeException("이메일 전송 실패");
        }
    }

    /**
     * 이메일 인증 코드 생성 및 Redis에 저장
     */
    public String generateAndSaveAuthCode(String email) {
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

        user.setEmailVerified(true);
        userRepository.save(user);

        redisUtil.delete(key);
        log.info("이메일 인증 성공 - email: {}", email);
    }
}
