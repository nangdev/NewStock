package newstock.domain.user.service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;
@Slf4j
@Service
@RequiredArgsConstructor
public class EmailSenderService {

    private final JavaMailSender javaMailSender;

    /**
     * 메일 전송
     */
    public void sendEmailCode(String email, String authCode) {
        MimeMessage message = javaMailSender.createMimeMessage();

        try {
            message.setFrom("team.newstock@gmail.com");
            message.setRecipients(MimeMessage.RecipientType.TO, email);
            message.setSubject("[NewStock] 이메일 인증 코드입니다.");

            String html = "<h3>요청하신 인증번호입니다.</h3>" +
                    "<h1 style='color:blue'>" + authCode + "</h1>" +
                    "<p>5분 이내로 입력해주세요.</p>";

            message.setText(html, "UTF-8", "html");

            javaMailSender.send(message);
            log.info("이메일 전송 및 인증 코드 저장 성공 - email: {}, code: {}", email, authCode);

        } catch (MessagingException e) {
            log.error("이메일 전송 실패 - email: {}", email, e);
            throw new ValidationException(ExceptionCode.EMAIL_SEND_FAILED);
        }
    }
}