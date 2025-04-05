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
        Optional<User> optionalUser = userRepository.findByEmail(userRequest.getEmail());

        if (optionalUser.isPresent()) {
            User existingUser = optionalUser.get();
            if (existingUser.isActivated()) {
                log.warn("중복된 이메일로 회원가입 시도 - email: {}", userRequest.getEmail());
                throw new ValidationException(ExceptionCode.DUPLICATE_EMAIL);
            }

            // 재가입
            existingUser.setPassword(passwordEncoder.encode(userRequest.getPassword()));
            existingUser.setNickname(userRequest.getNickname());
            existingUser.setRole((byte) 0);
            existingUser.setActivated(true);

            userRepository.save(existingUser);
            redisUtil.delete("email:verified:" + userRequest.getEmail());
            log.info("🔁 비활성화 유저 복구 완료 - userId: {}, email: {}", existingUser.getUserId(), existingUser.getEmail());
            return;
        }

        // ✅ 신규 가입 로직
        String encodedPassword = passwordEncoder.encode(userRequest.getPassword());

                // ⚠️ [임시 주석] 이메일 인증 우회 (테스트용)
//        if (!Boolean.TRUE.equals(redisUtil.get("email:verified:" + userRequest.getEmail(), Boolean.class))) {
//            throw new ValidationException(ExceptionCode.EMAIL_NOT_VERIFIED);
//        }
        User newUser = User.of(userRequest, encodedPassword);
        User savedUser = userRepository.save(newUser);

        redisUtil.delete("email:verified:" + userRequest.getEmail());
        log.info("회원가입 성공 - userId: {}, email: {}", savedUser.getUserId(), savedUser.getEmail());
    }
//        Optional<User> optionalUser = userRepository.findByEmail(userRequest.getEmail());
//            // 이메일 중복 체크
//            if (optionalUser.isPresent()) {
//                User existingUser = optionalUser.get();
//                if (existingUser.isActivated()) {
//                    log.warn("중복된 이메일로 회원가입 시도 - email: {}", userRequest.getEmail());
//                    throw new ValidationException(ExceptionCode.DUPLICATE_EMAIL);
//            }
//
//            // 탈퇴 처리한 기존 유저면 복구 처리
//            existingUser.setActivated(true);
//            existingUser.setPassword(passwordEncoder.encode(userRequest.getPassword()));
//            existingUser.setNickname(userRequest.getNickname());
//            existingUser.setRole((byte) 0);
//
//            userRepository.save(existingUser);
//            redisUtil.delete("email:verified:" + userRequest.getEmail());
//            log.info("비활성화 유저 복구 완료 - userId: {}, email: {}", existingUser.getUserId(), existingUser.getEmail());
//            return;
//        }
//
//        String encodedPassword = passwordEncoder.encode(userRequest.getPassword());
//
//        // ⚠️ [임시 주석] 이메일 인증 우회 (테스트용)
////        if (!Boolean.TRUE.equals(redisUtil.get("email:verified:" + userRequest.getEmail(), Boolean.class))) {
////            throw new ValidationException(ExceptionCode.EMAIL_NOT_VERIFIED);
////        }
//
//        User newUser = User.of(userRequest, encodedPassword);
//        User savedUser = userRepository.save(newUser);
//
//        redisUtil.delete("email:verified:" + userRequest.getEmail());
//        log.info("회원가입 성공 - userId: {}, email: {}", savedUser.getUserId(), savedUser.getEmail());
//    }

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

    // 회원 탈퇴
    @Override
    public void deleteUser(Integer userId, String accessToken) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ValidationException(ExceptionCode.USER_NOT_FOUND));

        if (!user.isActivated()) {
            throw new ValidationException(ExceptionCode.USER_ALREADY_DELETED);
        }

        user.setRefreshToken(null);
        user.setFcmToken(null);
        tokenBlacklistService.addToBlacklist(accessToken);

        user.setActivated(false);
        log.info("회원 탈퇴 처리 완료 - userId: {}", userId);
    }
}