package newstock.domain.user.repository;

import newstock.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Integer>, UserCustomRepository  {

    /**
     * 이메일을 기반으로 사용자 조회
     *
     * @param email 사용자의 이메일
     * @return 이메일이 일치하는 사용자 (Optional<User>)
     */
    Optional<User> findByEmail(String email);

    /**
     * 이메일 중복 체크
     *
     * @param email 중복 확인할 이메일
     * @return 이메일이 존재하면 true, 없으면 false
     */
    boolean existsByEmail(String email);

}
