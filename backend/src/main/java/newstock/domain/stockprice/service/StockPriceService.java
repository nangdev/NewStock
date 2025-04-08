package newstock.domain.stockprice.service;

import newstock.domain.stockprice.dto.StockPriceDto;

import java.util.List;

public interface StockPriceService {

    List<StockPriceDto> getLast30Days(Integer stockId);

    void addStockPrice(StockPriceDto stockPriceDto);
}
