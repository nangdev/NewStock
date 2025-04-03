package newstock.controller.request;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class EmailRequest {

    @NotBlank(message = "이메일 주소를 입력해주세요.")
    @Email(message = "올바른 이메일 형식을 입력해주세요.")
    @Schema(description = "인증 이메일 주소", example = "ssafy@gmail.com")
    private String email;
}
