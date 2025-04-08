package newstock.domain.news.repository;

import com.querydsl.core.types.OrderSpecifier;
import com.querydsl.core.types.dsl.Expressions;
import com.querydsl.core.types.dsl.NumberExpression;
import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.news.entity.News;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import static newstock.domain.news.entity.QNews.news;


@RequiredArgsConstructor
@Repository
public class NewsCustomRepositoryImpl implements NewsCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public Page<News> findNewsByStockIdOrderByScoreAbs(Integer stockId, Pageable pageable) {

        Long total = jpaQueryFactory
                .select(news.count())
                .from(news)
                .where(news.stockId.eq(stockId))
                .fetchOne();

        NumberExpression<Integer> absScore = Expressions.numberTemplate(Integer.class, "abs({0})", news.score);
        OrderSpecifier<?> orderSpecifier = absScore.desc();

        List<News> content = jpaQueryFactory
                .selectFrom(news)
                .where(news.stockId.eq(stockId))
                .orderBy(orderSpecifier)
                .offset(pageable.getOffset())
                .limit(pageable.getPageSize())
                .fetch();

        return new PageImpl<>(content, pageable, total != null ? total : 0);
    }

    @Override
    public Optional<List<News>> getTopNewsListByStockId(Integer stockId) {
        return Optional.ofNullable(jpaQueryFactory
                .selectFrom(news)
                .where(
                        news.stockId.eq(stockId)
                                .and(news.publishedDate.substring(0, 10).eq(LocalDate.now().toString()))
                )
                .orderBy(
                        Expressions.numberTemplate(Integer.class, "abs({0})", news.score).desc()
                )
                .limit(5)
                .fetch());
    }

    @Override
    public List<News> findNewsByStockIdAndDate(Integer stockId, String publishedDate) {
        return jpaQueryFactory.selectFrom(news)
                .where(
                        news.stockId.eq(stockId)
                                .and(news.publishedDate.startsWith(publishedDate))
                )
                .fetch();
    }
}