package newstock.domain.user.service;

import newstock.controller.response.UserResponse;

public interface UserService {

    UserResponse getUserById(int id);
}
