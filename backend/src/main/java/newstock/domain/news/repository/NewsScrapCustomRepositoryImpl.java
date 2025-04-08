package newstock.domain.news.repository;

import com.querydsl.core.types.OrderSpecifier;
import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.news.entity.News;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

import static newstock.domain.news.entity.QNews.news;
import static newstock.domain.news.entity.QNewsScrap.newsScrap;

@RequiredArgsConstructor
@Repository
public class NewsScrapCustomRepositoryImpl implements NewsScrapCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public Page<News> findScrappedNewsByUserIdAndStockId(Integer userId, Integer stockId, Pageable pageable) {

        Long total = jpaQueryFactory
                .select(news.count())
                .from(newsScrap)
                .join(news).on(news.newsId.eq(newsScrap.newsId))
                .where(newsScrap.userId.eq(userId)
                        .and(news.stockId.eq(stockId)))
                .fetchOne();

        OrderSpecifier<?> orderSpecifier = null;
        if (pageable.getSort().isSorted()) {
            for (Sort.Order order : pageable.getSort()) {
                if ("score".equalsIgnoreCase(order.getProperty())) {
                    orderSpecifier = order.isAscending() ? news.score.asc() : news.score.desc();
                } else if ("publishedDate".equalsIgnoreCase(order.getProperty())) {
                    orderSpecifier = order.isAscending() ? news.publishedDate.asc() : news.publishedDate.desc();
                }
            }
        }

        List<News> results = jpaQueryFactory
                .select(news)
                .from(newsScrap)
                .join(news).on(news.newsId.eq(newsScrap.newsId))
                .where(newsScrap.userId.eq(userId)
                        .and(news.stockId.eq(stockId)))
                .orderBy(orderSpecifier)
                .offset(pageable.getOffset())
                .limit(pageable.getPageSize())
                .fetch();

        return new PageImpl<>(results, pageable, total != null ? total : 0);
    }

    @Override
    public Optional<Integer> findIdByNewsIdAndUserId(Integer newsId, Integer userId) {

        return Optional.ofNullable(jpaQueryFactory
                .select(newsScrap.scrapId)
                .from(newsScrap)
                .where(newsScrap.newsId.eq(newsId)
                        .and(newsScrap.userId.eq(userId)))
                .fetchOne());
    }
}
