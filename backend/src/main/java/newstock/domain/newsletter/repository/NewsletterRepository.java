package newstock.domain.newsletter.repository;

import newstock.domain.newsletter.entity.Newsletter;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface NewsletterRepository extends JpaRepository<Newsletter, Integer>, NewsletterCustomRepository {

    Optional<Newsletter> findByStockIdAndDate(Integer stockId, String date);
}
