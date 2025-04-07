package newstock.domain.user.service;

import newstock.controller.request.LoginRequest;
import newstock.controller.response.LoginResponse;

public interface  AuthService {
    LoginResponse login(LoginRequest loginRequest);

    void logout(Integer userId, String accessToken);

    void clearFcmToken(Integer userId);

    LoginResponse reissueToken(String refreshToken, String fcmToken);

    LoginResponse loginWithKakao(String code, String fcmToken);
}
