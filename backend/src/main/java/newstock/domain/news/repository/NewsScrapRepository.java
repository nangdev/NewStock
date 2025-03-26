package newstock.domain.news.repository;

import newstock.domain.news.entity.NewsScrap;
import org.springframework.data.jpa.repository.JpaRepository;

public interface NewsScrapRepository extends JpaRepository<NewsScrap,Integer>, NewsScrapCustomRepository {

    boolean existsByNewsIdAndUserId(int newsId, int userId);
}
