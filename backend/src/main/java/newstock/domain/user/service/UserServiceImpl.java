package newstock.domain.user.service;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.UserRequest;
import newstock.controller.response.UserResponse;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
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
    public void addUser(UserRequest userRequest) {
        // 이메일 중복 체크
        if (userRepository.existsByEmail(userRequest.getEmail())) {
            log.warn("중복된 이메일로 회원가입 시도 - email: {}", userRequest.getEmail());
            throw new ValidationException(ExceptionCode.DUPLICATE_EMAIL);
        }

        String encodedPassword = passwordEncoder.encode(userRequest.getPassword());

        User newUser = User.of(userRequest, encodedPassword);
        User savedUser = userRepository.save(newUser);

        log.info("회원가입 성공 - userId: {}, email: {}", savedUser.getUserId(), savedUser.getEmail());
    }

    // 이메일 중복 체크 기능
    @Override
    public boolean existsByEmail(String email) {
        boolean exists = userRepository.existsByEmail(email);
        log.debug("이메일 중복 확인 - 이메일: {}, 존재 여부: {}", email, exists);

        return userRepository.existsByEmail(email);
    }
}
