package newstock.domain.news.repository;

import newstock.domain.news.entity.News;
import org.springframework.data.domain.*;

public interface NewsScrapCustomRepository {
    Page<News> findScrappedNewsByUserIdAndStockCode(int userId, int stockCode, Pageable pageable);
}
