package newstock.domain.stock.repository;

import newstock.domain.stock.entity.Stock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface StockRepository extends JpaRepository<Stock, Integer> , StockRepositoryCustom {
    Stock findStocksByStockCode(int stockCode);
}
