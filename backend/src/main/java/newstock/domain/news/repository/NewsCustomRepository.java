package newstock.domain.news.repository;

import newstock.domain.news.entity.News;

import java.util.List;
import java.util.Optional;

public interface NewsCustomRepository {

    public Optional<List<News>> getTopNewsByStockCode(int stockCode);
}
