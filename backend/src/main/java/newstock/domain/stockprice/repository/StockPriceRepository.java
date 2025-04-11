package newstock.domain.stockprice.repository;

import newstock.domain.stockprice.entity.StockPrice;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface StockPriceRepository extends JpaRepository<StockPrice, Integer> {

    List<StockPrice> findByStockIdOrderByDateAsc(Integer stockId);
}
