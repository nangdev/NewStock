package newstock.domain.newsletter.repository;

import newstock.domain.newsletter.entity.Keyword;
import org.springframework.data.jpa.repository.JpaRepository;

public interface KeywordRepository extends JpaRepository<Keyword, Integer>, KeywordCustomRepository {
}
