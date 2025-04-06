package newstock.domain.user.repository;

import newstock.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Integer>, UserCustomRepository  {

    // 이메일로 전체 사용자 조회
    Optional<User> findByEmail(String email);

    // 아이디로 활성화된 사용자 조회
    Optional<User> findByUserIdAndActivatedTrue(Integer userId);

    // 이메일로 활성화된 사용자 조회
    Optional<User> findByEmailAndActivatedTrue(String email);

    // 카카오 ID로 활성화된 사용자 조회
    Optional<User> findByKakaoIdAndActivatedTrue(Long kakaoId);

    // 활성화된 이메일이 존재하는지 여부
    boolean existsByEmailAndActivatedTrue(String email);
}
