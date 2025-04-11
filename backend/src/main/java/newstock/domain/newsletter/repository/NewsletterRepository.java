package newstock.domain.newsletter.repository;

import newstock.domain.newsletter.entity.Newsletter;
import org.springframework.data.jpa.repository.JpaRepository;

public interface NewsletterRepository extends JpaRepository<Newsletter, Integer>, NewsletterCustomRepository {

    boolean existsByDate(String date);

}