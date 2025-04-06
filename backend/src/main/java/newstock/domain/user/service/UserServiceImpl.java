package newstock.domain.user.service;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.jwt.TokenBlacklistService;
import newstock.common.redis.RedisUtil;
import newstock.controller.request.UserRequest;
import newstock.controller.response.UserResponse;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final RedisUtil redisUtil;
    private final TokenBlacklistService tokenBlacklistService;

    // 회원 가입
    @Override
    @Transactional
    public void addUser(UserRequest userRequest) {
        String email = userRequest.getEmail();
        String encodedPassword = passwordEncoder.encode(userRequest.getPassword());

        Optional<User> existingUser = userRepository.findByEmail(userRequest.getEmail());
        if (existingUser.isPresent()) {
            User user = existingUser.get();
            log.info("회원 가입 시도 이메일 - email: {}", user.getEmail());

            // 재가입일 경우 복구 처리
            if (!user.isActivated()) {
                user.reactivate(userRequest, encodedPassword);
                userRepository.save(user);
                log.info("탈퇴 유저 복구 - userId: {}, email: {}", user.getUserId(), user.getEmail());
                return;
            } else {
                log.warn("중복된 이메일로 회원가입 시도 - email: {}",  email);
                throw new ValidationException(ExceptionCode.DUPLICATE_EMAIL);
            }
        }
        // ⚠️ [임시 주석] 이메일 인증 우회 (테스트용)
//        if (!Boolean.TRUE.equals(redisUtil.get("email:verified:" + userRequest.getEmail(), Boolean.class))) {
//            throw new ValidationException(ExceptionCode.EMAIL_NOT_VERIFIED);
//        }

        // 신규 가입
        User newUser = User.of(userRequest, encodedPassword);
        userRepository.save(newUser);
//      ⚠️ [임시 주석]
//   redisUtil.delete("email:verified:" + userRequest.getEmail());
        log.info("회원 가입 완료 - userId: {}, email: {}", newUser.getUserId(), newUser.getEmail());
    }

    // 이메일 중복 체크
    @Override
    public boolean existsByEmail(String email) {
        boolean exists = userRepository.existsByEmailAndIsActivatedTrue(email);
        log.debug("이메일 중복 확인 - 이메일: {}, 존재 여부: {}", email, exists);

        return exists;
    }

    // 회원가입 후 최초 로그인 시, 유저 권한을 1(USER)로 변경
    @Override
    @Transactional
    public void updateUserRole(Integer userId) {
        User user = userRepository.findByUserIdAndIsActivatedTrue(userId)
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
        User user = userRepository.findByUserIdAndIsActivatedTrue(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        return UserResponse.of(user);
    }

    // 회원 탈퇴
    @Override
    public void deleteUser(Integer userId, String accessToken) {
        User user = userRepository.findByUserIdAndIsActivatedTrue(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        if (!user.isActivated()) {
            throw new ValidationException(ExceptionCode.USER_ALREADY_DELETED);
        }

        user.setRefreshToken(null);
        user.setFcmToken(null);
        tokenBlacklistService.addToBlacklist(accessToken);
        user.setActivated(false);
        userRepository.save(user);
        log.info("회원 탈퇴 처리 완료 - userId: {}", userId);
    }
}