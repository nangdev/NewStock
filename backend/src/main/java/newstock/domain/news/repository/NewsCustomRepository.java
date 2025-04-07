package newstock.domain.news.repository;

import newstock.domain.news.entity.News;

import java.util.List;
import java.util.Optional;

public interface NewsCustomRepository {

    Optional<List<News>> getTopNewsListByStockId(Integer stockId);

    List<News> findNewsByStockIdAndDate(Integer stockId, String publishedDate);
}
