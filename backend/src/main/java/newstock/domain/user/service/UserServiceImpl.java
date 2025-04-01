package newstock.domain.user.service;

import io.jsonwebtoken.Claims;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.jwt.TokenBlacklistService;
import newstock.controller.request.LoginRequest;
import newstock.controller.request.UserRequest;
import newstock.controller.response.LoginResponse;
import newstock.controller.response.UserResponse;
import newstock.domain.user.dto.JwtToken;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import newstock.domain.user.service.CustomUserDetails;
import newstock.common.jwt.JwtTokenProvider;

import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;
    private final TokenBlacklistService tokenBlacklistService;

    // 회원 가입
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

    // 이메일 중복 체크
    @Override
    public boolean existsByEmail(String email) {
        boolean exists = userRepository.existsByEmail(email);
        log.debug("이메일 중복 확인 - 이메일: {}, 존재 여부: {}", email, exists);

        return exists;
    }

    // 회원가입 후 최초 로그인 시, 유저 권한을 1(USER)로 변경
    @Override
    @Transactional
    public void updateUserRole(Integer userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        if (user.getRole() != 0) {
            throw new ValidationException(ExceptionCode.USER_ROLE_UPDATE_ERROR);
        }

        user.setRole((byte) 1);
        userRepository.save(user);
    }

    // 유저 정보 조회
    @Override
    public UserResponse getUserInfo(Integer userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        return UserResponse.of(user);
    }
}