package newstock.controller.request;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class EmailVerifyRequest {

    @NotBlank(message = "이메일 주소를 입력해주세요.")
    @Email(message = "올바른 이메일 형식을 입력해주세요.")
    @Schema(description = "이메일", example = "ssafy@gmail.com")
    private String email;

    @NotBlank(message = "인증 코드를 입력해주세요.")
    @Schema(description = "6자리 인증 코드", example = "333333")
    private String authCode;
}