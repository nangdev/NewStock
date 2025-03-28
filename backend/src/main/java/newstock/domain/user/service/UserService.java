package newstock.domain.user.service;

import newstock.controller.request.LoginRequest;
import newstock.controller.request.UserRequest;
import newstock.controller.response.LoginResponse;
import newstock.controller.response.UserResponse;
import newstock.domain.user.entity.User;

public interface UserService {
    // 회원 정보 관련
    void addUser(UserRequest userRequest);

    boolean existsByEmail(String email);

    void updateUserRole(Integer userId);

    UserResponse getUserInfo(Integer userId);

    // 인증 관련
    LoginResponse login(LoginRequest loginRequest);

    void logout(Integer userId, String accessToken);

    void clearFcmToken(Integer userId);
}
