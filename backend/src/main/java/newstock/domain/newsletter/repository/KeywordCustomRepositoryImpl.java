package newstock.domain.newsletter.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

@Repository
@RequiredArgsConstructor
public class KeywordCustomRepositoryImpl implements KeywordCustomRepository {

    private final JPAQueryFactory queryFactory;
}
