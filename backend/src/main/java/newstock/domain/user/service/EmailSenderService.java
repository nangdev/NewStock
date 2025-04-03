package newstock.domain.user.service;


import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.core.io.ClassPathResource;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmailSenderService {

    private final JavaMailSender javaMailSender;
    private static final String TEMPLATE_PATH = "templates/email/verification.html";

    /**
     * 메일 전송
     */
    public void sendEmailCode(String email, String authCode) {
        MimeMessage message = javaMailSender.createMimeMessage();
        String username = email.split("@")[0];

        try {
            String html = loadHtmlTemplate(TEMPLATE_PATH, username, authCode);

            message.setFrom("NewStock <team.newstock@gmail.com>");
            message.setRecipients(MimeMessage.RecipientType.TO, email);

            message.setHeader("User-Agent", "NewStock-Mail-Client");
            message.setHeader("X-Mailer", "NewStockMailer 1.0");
            message.setHeader("Message-ID", "<" + UUID.randomUUID() + "@newstock.com>");

            message.setSubject("[NewStock] 이메일 인증 코드입니다.");
            message.setText(html, "UTF-8", "html");

            javaMailSender.send(message);
            log.info("이메일 전송 및 인증 코드 저장 성공 - email: {}, code: {}", email, authCode);

        } catch (MessagingException e) {
            log.error("이메일 전송 실패 - email: {}", email, e);
            throw new ValidationException(ExceptionCode.EMAIL_SEND_FAILED);
        }
    }

    private String loadHtmlTemplate(String path, String username, String authCode) {
        try {
            ClassPathResource resource = new ClassPathResource(path);
        String html = Files.readString(resource.getFile().toPath(), StandardCharsets.UTF_8);
        return html.replace("{{username}}", username)
                .replace("{{authCode}}", authCode);
        } catch (IOException e) {
            log.error("이메일 템플릿 로드 실패", e);
            throw new ValidationException(ExceptionCode.EMAIL_SEND_FAILED);
        }
    }
}