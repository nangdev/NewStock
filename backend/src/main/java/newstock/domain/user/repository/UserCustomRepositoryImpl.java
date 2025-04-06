package newstock.domain.user.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import jakarta.persistence.EntityManager;
import newstock.domain.user.entity.User;
import static newstock.domain.user.entity.QUser.user;
import org.springframework.stereotype.Repository;

import java.util.Optional;


@Repository
public class UserCustomRepositoryImpl implements UserCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    public UserCustomRepositoryImpl(EntityManager entityManager) {
        this.jpaQueryFactory = new JPAQueryFactory(entityManager);
    }

    // 활성화된 유저만 ID 기준으로 조회
    @Override
    public Optional<User> findActivatedById(Integer userId) {
        return Optional.ofNullable(
                jpaQueryFactory
                        .selectFrom(user)
                        .where(user.userId.eq(userId),
                                user.isActivated.isTrue())
                        .fetchOne()
        );
    }
}