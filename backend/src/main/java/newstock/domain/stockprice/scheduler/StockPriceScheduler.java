package newstock.domain.stockprice.scheduler;

import lombok.RequiredArgsConstructor;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.service.StockService;
import newstock.domain.stockprice.dto.StockPriceDto;
import newstock.domain.stockprice.service.StockPriceInfoService;
import newstock.domain.stockprice.util.NaverFinanceCrawler;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
public class StockPriceScheduler {

    private final StockPriceInfoService stockPriceInfoService;

    private final StockService stockService;

    private final NaverFinanceCrawler naverFinanceCrawler;

    @Scheduled(cron = "0 0 16 * * MON-FRI")
    public void updateTodayStockData() throws Exception {
        List<StockDto> stockDtoList = stockService.getAllStockList();
        for (StockDto stockDto : stockDtoList) {

            StockPriceDto stockPriceDto = naverFinanceCrawler.getLatestStockPrice(stockDto.getStockId());

            stockPriceInfoService.addStockPrice(stockPriceDto);
        }
    }
}
