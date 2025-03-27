package newstock.domain.news.repository;

import newstock.domain.news.entity.News;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface NewsRepository extends JpaRepository<News, Integer>, NewsCustomRepository {
    Page<News> findByStockId(Integer stockId, Pageable pageable);
}
