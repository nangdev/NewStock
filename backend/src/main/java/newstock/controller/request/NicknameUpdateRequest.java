package newstock.controller.request;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;

@Getter
public class NicknameUpdateRequest {

    @NotBlank(message = "변경할 닉네임을 입력해주세요.")
    @Size(min = 2, max = 10, message = "닉네임은 2자 이상 10자 이하로 입력해주세요.")
    @Schema(description = "새로운 닉네임", example = "뉴싸피")
    private String nickname;
}