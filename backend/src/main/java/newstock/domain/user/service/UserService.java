package newstock.domain.user.service;

import newstock.controller.request.UserRequest;
import newstock.controller.response.UserResponse;

public interface UserService {

    // 회원가입
    UserResponse signupUser(UserRequest userRequest);
}
