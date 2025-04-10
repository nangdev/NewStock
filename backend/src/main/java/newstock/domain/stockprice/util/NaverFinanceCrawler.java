package newstock.domain.stockprice.util;

import lombok.RequiredArgsConstructor;
import newstock.domain.stock.service.StockService;
import newstock.domain.stockprice.dto.StockPriceDto;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

@Component
@RequiredArgsConstructor
public class NaverFinanceCrawler {

    private final StockService stockService;

    public StockPriceDto getLatestStockPrice(Integer stockId) throws Exception {

        String stockCode = stockService.getStockInfo(stockId).getStockCode();
        String url = "https://finance.naver.com/item/sise_day.nhn?code=" + stockCode + "&page=1";
        Document doc = Jsoup.connect(url)
                .timeout(5000)
                .get();

        Elements rows = doc.select("table.type2 tr:has(td)");
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd");

        for (Element row : rows) {
            Elements tds = row.select("td");
            if (tds.size() >= 2) {
                String dateStr = tds.get(0).text().trim();
                String closeStr = tds.get(1).text().trim().replaceAll(",", "");
                if (!dateStr.isEmpty() && !closeStr.isEmpty()) {
                    try {
                        LocalDate date = LocalDate.parse(dateStr, formatter);
                        int price = Integer.parseInt(closeStr);
                        return StockPriceDto.builder()
                                .stockId(stockId)
                                .date(date)
                                .price(price)
                                .build();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        return null;
    }
}
