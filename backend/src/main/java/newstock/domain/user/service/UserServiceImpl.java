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

    // 로그아웃
    @Override
    @Transactional
    public void logout(Integer userId, String accessToken) {
        tokenBlacklistService.addToBlacklist(accessToken);
        log.info("토큰 블랙리스트 등록 완료 - userId: {}, token: {}", userId, accessToken);

        clearFcmToken(userId);
    }

    // FCM 토큰 초기화
    @Override
    @Transactional
    public void clearFcmToken(Integer userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        user.setFcmToken(null);
        userRepository.save(user);

        log.info("FCM 토큰 초기화 완료 - userId: {}", userId);
    }

    // JWT 토큰 재발급
    @Override
    @Transactional
    public LoginResponse reissueToken(String refreshToken, String fcmToken) {
        Integer userId = jwtTokenProvider.getUserIdFromRefreshToken(refreshToken);

        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        if (!refreshToken.equals(user.getRefreshToken())) {
            throw new ValidationException(ExceptionCode.TOKEN_INVALID, "refreshToken이 일치하지 않습니다.");
        }

        CustomUserDetails userDetails = new CustomUserDetails(user);
        Authentication authentication = new UsernamePasswordAuthenticationToken(
                userDetails, null, userDetails.getAuthorities());

        JwtToken newToken = jwtTokenProvider.generateToken(authentication);

        user.setRefreshToken(newToken.getRefreshToken());
        user.setFcmToken(fcmToken);

        userRepository.save(user);

        return LoginResponse.builder()
                .accessToken(newToken.getAccessToken())
                .refreshToken(newToken.getRefreshToken())
                .build();
    }
}