package newstock.domain.stock.service;

import newstock.external.kis.dto.KisRealTimeStockPriceDto;

public interface StockPriceService {
    void sendMessage(String message);

    void sendStockInfo(KisRealTimeStockPriceDto RTStockInfoDto);
}
