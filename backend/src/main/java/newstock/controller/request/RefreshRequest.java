package newstock.controller.request;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class RefreshRequest {
    @Schema(description = "FCM 토큰", example = "fcm_token_abc123")
    @NotBlank(message = "FCM 토큰은 필수입니다.")
    private String fcmToken;
}
