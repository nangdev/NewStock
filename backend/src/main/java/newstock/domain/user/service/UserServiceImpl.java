package newstock.domain.user.service;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.LoginRequest;
import newstock.controller.request.UserRequest;
import newstock.controller.response.LoginResponse;
import newstock.controller.response.UserResponse;
import newstock.domain.user.dto.JwtToken;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import newstock.common.jwt.JwtTokenProvider;

import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import org.springframework.security.core.Authentication;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;

    @Override
    @Transactional
    public UserResponse signupUser(UserRequest userRequest) {
        // 이메일 중복 체크
        if (userRepository.existsByEmail(userRequest.getEmail())) {
            throw new ValidationException(ExceptionCode.DUPLICATE_EMAIL);
        }

        // 비밀번호 해싱
        String encodedPassword = passwordEncoder.encode(userRequest.getPassword());

        // User 엔티티 생성 및 저장
        User newUser = User.of(userRequest, encodedPassword);
        User savedUser = userRepository.save(newUser);

        // UserResponse 반환
        return UserResponse.of(savedUser);
    }

    // 이메일 중복 체크 기능
    @Override
    public boolean existsByEmail(String email) {
        return userRepository.existsByEmail(email);
    }

    // 로그인
    @Override
    @Transactional
    public LoginResponse login(LoginRequest request) {
        User user = userRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> new ValidationException(ExceptionCode.VALIDATION_ERROR, "이메일 또는 비밀번호가 일치하지 않습니다."));

        if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
            throw new ValidationException(ExceptionCode.VALIDATION_ERROR, "이메일 또는 비밀번호가 일치하지 않습니다.");
        }

        CustomUserDetails userDetails = new CustomUserDetails(user);
        Authentication authentication = new UsernamePasswordAuthenticationToken(
                userDetails, null, userDetails.getAuthorities());

        JwtToken token = jwtTokenProvider.generateToken(authentication);

        user.setAccessToken(token.getAccessToken());
        user.setRefreshToken(token.getRefreshToken());

        if (request.getFcmToken() != null && !request.getFcmToken().isBlank()) {
            user.setFcmToken(request.getFcmToken());
        }
        userRepository.save(user);

        return LoginResponse.builder()
                .accessToken(token.getAccessToken())
                .refreshToken(token.getRefreshToken())
                .build();
    }
}

