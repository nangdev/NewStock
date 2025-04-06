package newstock.domain.user.service;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.jwt.JwtTokenProvider;
import newstock.common.jwt.TokenBlacklistService;
import newstock.controller.request.LoginRequest;
import newstock.controller.response.LoginResponse;
import newstock.domain.user.dto.JwtToken;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import newstock.external.kakao.KakaoOauthService;
import newstock.external.kakao.dto.KakaoTokenResponse;
import newstock.external.kakao.dto.KakaoUserInfo;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthServiceImpl implements AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;
    private final TokenBlacklistService tokenBlacklistService;
    private final KakaoOauthService kakaoOauthService;


    // 로그인
    @Override
    @Transactional
    public LoginResponse login(LoginRequest request) {
        User user = userRepository.findByEmailAndIsActivatedTrue(request.getEmail())
                .orElseThrow(() -> new ValidationException(ExceptionCode.VALIDATION_ERROR));

        if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
            throw new ValidationException(ExceptionCode.VALIDATION_ERROR);
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
        User user = userRepository.findByUserIdAndIsActivatedTrue(userId)
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

        User user = userRepository.findByUserIdAndIsActivatedTrue(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        if (!refreshToken.equals(user.getRefreshToken())) {
            throw new ValidationException(ExceptionCode.TOKEN_INVALID);
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

    @Override
    @Transactional
    public LoginResponse loginWithKakao(String code, String fcmToken) {

        KakaoTokenResponse tokenResponse = kakaoOauthService.getToken(code);
        String kakaoAccessToken = tokenResponse.getAccessToken();

        KakaoUserInfo userInfo = kakaoOauthService.getUserInfo(kakaoAccessToken);
        Long kakaoId = userInfo.getId();
        String email = userInfo.getKakaoAccount().getEmail();
        String nickname = userInfo.getKakaoAccount().getProfile().getNickname();

        User user = userRepository.findByKakaoIdAndIsActivatedTrue(kakaoId)
                .orElseGet(() -> addNewKakaoUser(kakaoId, email, nickname)); // 없으면 회원가입

        CustomUserDetails userDetails = new CustomUserDetails(user);
        Authentication authentication = new UsernamePasswordAuthenticationToken(
                userDetails, null, userDetails.getAuthorities());

        JwtToken jwt = jwtTokenProvider.generateToken(authentication);

        user.setRefreshToken(jwt.getRefreshToken());
        if (fcmToken != null && !fcmToken.isBlank()) {
            user.setFcmToken(fcmToken);
        }

        userRepository.save(user);

        return LoginResponse.builder()
                .accessToken(jwt.getAccessToken())
                .refreshToken(jwt.getRefreshToken())
                .build();
    }

    private User addNewKakaoUser(Long kakaoId, String email, String nickname) {
        User user = User.ofKakao(kakaoId, email, nickname);

        return userRepository.save(user);
    }
}