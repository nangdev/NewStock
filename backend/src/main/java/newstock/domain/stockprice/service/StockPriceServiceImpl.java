package newstock.domain.stockprice.service;

import lombok.RequiredArgsConstructor;
import newstock.controller.response.StockPriceResponse;
import newstock.domain.stockprice.dto.StockPriceDto;
import newstock.domain.stockprice.entity.StockPrice;
import newstock.domain.stockprice.repository.StockPriceRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class StockPriceServiceImpl implements StockPriceService {

    private final StockPriceRepository stockPriceRepository;

    public StockPriceResponse getAllStockPrices(Integer stockId) {
        return StockPriceResponse.of(stockPriceRepository.findByStockIdOrderByDateAsc(stockId)
                .stream()
                .map(StockPriceDto::of)
                .collect(Collectors.toList()));
    }

    public void addStockPrice(StockPriceDto stockPriceDto) {
        // 새로운 데이터를 저장
        StockPrice newPrice = stockPriceDto.toEntity();
        stockPriceRepository.save(newPrice);

        // 해당 종목에 대한 모든 데이터를 날짜 오름차순으로 조회
        List<StockPrice> prices = stockPriceRepository.findByStockIdOrderByDateAsc(newPrice.getStockId());

        // 전체 개수가 30개보다 초과하면 가장 오래된(날짜가 가장 낮은, 첫번째) 데이터를 삭제
        if (prices.size() > 30) {
            StockPrice oldest = prices.get(0);
            stockPriceRepository.delete(oldest);
        }
    }
}
