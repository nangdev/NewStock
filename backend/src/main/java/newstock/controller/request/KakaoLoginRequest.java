package newstock.controller.request;

import lombok.Data;

@Data
public class KakaoLoginRequest {
    private String code;
    private String fcmToken;
}
