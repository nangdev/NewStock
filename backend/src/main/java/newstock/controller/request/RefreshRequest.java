package newstock.controller.request;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
public class RefreshRequest {
    @Schema(description = "FCM 토큰", example = "fcm_token_abc123")
    private String fcmToken;
}
