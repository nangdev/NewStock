package newstock.domain.keyword.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.keyword.entity.Keyword;
import org.springframework.stereotype.Repository;
import static newstock.domain.newsletter.entity.QKeyword.keyword;

import java.util.List;

@Repository
@RequiredArgsConstructor
public class KeywordCustomRepositoryImpl implements KeywordCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public List<Keyword> findByStockIdAndDate(Integer stockId, String date) {

        return jpaQueryFactory.selectFrom(keyword)
                .where(
                        keyword.stockId.eq(stockId)
                                .and(keyword.date.eq(date))
                )
                .orderBy(keyword.count.desc())
                .fetch();
    }
}
