package newstock.domain.user.service;

import newstock.controller.request.LoginRequest;
import newstock.controller.request.UserRequest;
import newstock.controller.response.LoginResponse;
import newstock.controller.response.UserResponse;

public interface UserService {

    // 회원가입
    UserResponse signupUser(UserRequest userRequest);

    // 이메일 중복체크
    boolean existsByEmail(String email);

    // 로그인
    LoginResponse login(LoginRequest loginRequest);
}
