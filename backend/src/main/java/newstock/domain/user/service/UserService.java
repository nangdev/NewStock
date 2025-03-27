package newstock.domain.user.service;

import newstock.controller.request.UserRequest;
import newstock.controller.response.UserResponse;

public interface UserService {

    // 회원가입
    void addUser(UserRequest userRequest);

    // 이메일 중복체크
    boolean existsByEmail(String email);

}
