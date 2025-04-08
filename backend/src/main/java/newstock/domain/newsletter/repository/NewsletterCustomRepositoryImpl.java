package newstock.domain.newsletter.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.newsletter.entity.Newsletter;
import org.springframework.stereotype.Repository;

import java.util.Optional;

import static newstock.domain.newsletter.entity.QNewsletter.newsletter;

@Repository
@RequiredArgsConstructor
public class NewsletterCustomRepositoryImpl implements NewsletterCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public Optional<Newsletter> findByStockIdAndDate(Integer stockId, String date) {

        return Optional.ofNullable(
                jpaQueryFactory
                        .selectFrom(newsletter)
                        .where(
                                newsletter.stockId.eq(stockId)
                                        .and(newsletter.date.eq(date))
                        )
                        .fetchOne());
    }
}
