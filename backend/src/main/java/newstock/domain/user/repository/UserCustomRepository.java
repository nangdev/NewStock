package newstock.domain.user.repository;

import newstock.domain.user.entity.User;

import java.util.Optional;

public interface UserCustomRepository {

    // 활성화된 유저만 ID 기준으로 조회
    Optional<User> findActivatedById(Integer userId);
}