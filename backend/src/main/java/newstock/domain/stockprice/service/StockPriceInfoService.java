package newstock.domain.stockprice.service;

import newstock.controller.response.StockPriceResponse;
import newstock.domain.stockprice.dto.StockPriceDto;

public interface StockPriceInfoService {

    StockPriceResponse getAllStockPrices(Integer stockId);

    void addStockPrice(StockPriceDto stockPriceDto);
}
