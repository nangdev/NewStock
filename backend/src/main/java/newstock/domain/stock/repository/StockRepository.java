package newstock.domain.stock.repository;

import newstock.domain.stock.dto.StockCodeDto;
import newstock.domain.stock.entity.Stock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface StockRepository extends JpaRepository<Stock, Integer>, StockRepositoryCustom {

    @Query("select s.stockCode from Stock s")
    List<String> findCodesAll();

    @Query("select new newstock.domain.stock.dto.StockCodeDto(s.stockId, s.stockCode) FROM Stock s")
    List<StockCodeDto> findAllStockCodes();

}
