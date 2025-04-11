package newstock.domain.newsletter.repository;

import newstock.domain.newsletter.entity.Newsletter;

import java.util.Optional;

public interface NewsletterCustomRepository {

    Optional<Newsletter> findByStockIdAndDate(Integer stockId, String date);

}
