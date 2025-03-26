package newstock.domain.news.repository;

import com.querydsl.core.types.OrderSpecifier;
import com.querydsl.jpa.impl.JPAQueryFactory;
import jakarta.persistence.EntityManager;
import newstock.domain.news.entity.News;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import static newstock.domain.news.entity.QNews.news;
import static newstock.domain.news.entity.QNewsScrap.newsScrap;

import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class NewsScrapCustomRepositoryImpl implements NewsScrapCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    public NewsScrapCustomRepositoryImpl(EntityManager entityManager) {
        this.jpaQueryFactory = new JPAQueryFactory(entityManager);
    }

    @Override
    public Page<News> findScrappedNewsByUserIdAndStockCode(int userId, int stockCode, Pageable pageable) {

        Long total = jpaQueryFactory
                .select(news.count())
                .from(newsScrap)
                .join(news).on(news.newsId.eq(newsScrap.newsId))
                .where(newsScrap.userId.eq(userId)
                        .and(news.stockCode.eq(stockCode)))
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
                        .and(news.stockCode.eq(stockCode)))
                .orderBy(orderSpecifier)
                .offset(pageable.getOffset())
                .limit(pageable.getPageSize())
                .fetch();

        return new PageImpl<>(results, pageable, total != null ? total : 0);
    }
}
