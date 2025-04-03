package newstock.domain.news.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.news.entity.News;
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
    public Optional<List<News>> getTopNewsListByStockId(Integer stockId) {
        return Optional.ofNullable(jpaQueryFactory
                .selectFrom(news)
                .where(
                        news.stockId.eq(stockId)
                                .and(news.publishedDate.substring(0, 10).eq(LocalDate.now().toString()))
                )
                .orderBy(news.score.desc())
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
