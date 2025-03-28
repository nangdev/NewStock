package newstock.domain.stock.service;

import newstock.external.kis.dto.KisStockInfoDto;

public interface StockPriceService {
    void sendMessage(String message);
    void sendStockInfo(KisStockInfoDto stockInfoDto);
}
