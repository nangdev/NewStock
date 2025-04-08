package newstock.domain.news.repository;

import newstock.domain.news.entity.News;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import java.util.List;
import java.util.Optional;

public interface NewsCustomRepository {

    Page<News> findNewsByStockIdOrderByScoreAbs(Integer stockId, Pageable pageable);

    Optional<List<News>> getTopNewsListByStockId(Integer stockId);

    List<News> findNewsByStockIdAndDate(Integer stockId, String publishedDate);
}
