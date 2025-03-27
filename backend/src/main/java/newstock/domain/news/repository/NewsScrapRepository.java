package newstock.domain.news.repository;

import newstock.domain.news.entity.NewsScrap;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface NewsScrapRepository extends JpaRepository<NewsScrap,Integer>, NewsScrapCustomRepository {

    boolean existsByNewsIdAndUserId(int newsId, int userId);

    Optional<Integer> findIdByNewsIdAndUserId(int newsId, int userId);
}
