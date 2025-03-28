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
}
