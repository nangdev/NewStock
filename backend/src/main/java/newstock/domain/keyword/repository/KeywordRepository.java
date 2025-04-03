package newstock.domain.keyword.repository;

import newstock.domain.keyword.entity.Keyword;
import org.springframework.data.jpa.repository.JpaRepository;

public interface KeywordRepository extends JpaRepository<Keyword, Integer>, KeywordCustomRepository {
}
