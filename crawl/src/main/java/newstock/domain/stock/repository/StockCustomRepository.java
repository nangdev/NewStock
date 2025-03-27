package newstock.domain.stock.repository;

import newstock.domain.stock.dto.StockDto;

import java.util.List;

public interface StockCustomRepository {

    List<StockDto> findAllStockIdAndName();
}
