package newstock.domain.stock.service;

import newstock.domain.stock.dto.StockDto;

import java.util.List;

public interface StockService {

    List<StockDto> getAllStocks();
}
