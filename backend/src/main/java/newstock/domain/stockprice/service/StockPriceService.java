package newstock.domain.stockprice.service;

import newstock.controller.response.StockPriceResponse;
import newstock.domain.stockprice.dto.StockPriceDto;

import java.util.List;

public interface StockPriceService {

    StockPriceResponse getAllStockPrices(Integer stockId);

    void addStockPrice(StockPriceDto stockPriceDto);
}
