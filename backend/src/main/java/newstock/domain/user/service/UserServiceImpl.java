package newstock.domain.user.service;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.UserRequest;
import newstock.controller.response.UserResponse;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;


    @Override
    @Transactional
    public UserResponse signupUser(UserRequest userRequest) {

        // 비밀번호 해싱
        String encodedPassword = passwordEncoder.encode(userRequest.getPassword());

        // User 엔티티 생성 및 저장
        User newUser = User.builder()
                .email(userRequest.getEmail())
                .password(encodedPassword)
                .username(userRequest.getUsername())
                .nickname(userRequest.getNickname())
                .build();

        User savedUser = userRepository.save(newUser);

        // UserResponse 반환
        return UserResponse.builder()
                .userId(savedUser.getUserId())
                .email(savedUser.getEmail())
                .username(savedUser.getUsername())
                .nickname(savedUser.getNickname())
                .build();
    }
}
