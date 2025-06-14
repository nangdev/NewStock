package newstock.domain.news.repository;

import newstock.domain.news.entity.News;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import java.util.Optional;

public interface NewsScrapCustomRepository {

    Page<News> findScrappedNewsByUserIdAndStockId(Integer userId, Integer stockId, Pageable pageable);

    Optional<Integer> findIdByNewsIdAndUserId(Integer newsId, Integer userId);
}
