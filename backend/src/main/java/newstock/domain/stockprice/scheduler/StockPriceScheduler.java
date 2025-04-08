package newstock.domain.stockprice.scheduler;

import lombok.RequiredArgsConstructor;
import newstock.domain.stockprice.service.StockPriceService;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class StockPriceScheduler {

    private final StockPriceService stockPriceService;

    @Scheduled(cron = "0 0 18 * * MON-FRI")
    public void updateTodayStockData() {

    }
}
