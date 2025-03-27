package newstock.domain.user.service;

import newstock.controller.request.LoginRequest;
import newstock.controller.request.UserRequest;
import newstock.controller.response.LoginResponse;
import newstock.controller.response.UserResponse;
import newstock.domain.user.entity.User;

public interface UserService {

    // 회원가입
    void addUser(UserRequest userRequest);

    // 이메일 중복체크
    boolean existsByEmail(String email);

    // 로그인
    LoginResponse login(LoginRequest loginRequest);

    void updateUserRole(User user);
}
