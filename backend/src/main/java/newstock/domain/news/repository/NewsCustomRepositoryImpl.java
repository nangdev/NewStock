package newstock.domain.news.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import jakarta.persistence.EntityManager;
import lombok.RequiredArgsConstructor;
import newstock.domain.news.entity.News;
import static newstock.domain.news.entity.QNews.news;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;


@RequiredArgsConstructor
@Repository
public class NewsCustomRepositoryImpl implements NewsCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public Optional<List<News>> getTopNewsListByStockCode(int stockCode) {
        return Optional.ofNullable(jpaQueryFactory
                .selectFrom(news)
                .where(
                        news.stockCode.eq(stockCode)
                                .and(news.publishedDate.substring(0, 10).eq(LocalDate.now().toString()))
                )
                .orderBy(news.score.desc())
                .limit(5)
                .fetch());

    }
}
