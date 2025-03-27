package newstock.domain.stock.repository;

import newstock.domain.stock.entity.Stock;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StockRepository extends JpaRepository<Stock, Integer>, StockCustomRepository {
}
