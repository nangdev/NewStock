package newstock.domain.stockprice.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.stockprice.dto.StockPriceDto;
import newstock.domain.stockprice.entity.StockPrice;
import newstock.domain.stockprice.repository.StockPriceRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class StockPriceServiceImpl implements StockPriceService {

   private final StockPriceRepository stockPriceRepository;

    @Override
    public List<StockPriceDto> getLast30Days(Integer stockId) {

        LocalDate today = LocalDate.now();

        LocalDate startDate = today.minusDays(30);

        return stockPriceRepository.findByStockIdAndDateBetweenOrderByDateAsc(stockId,startDate,today)
                .stream()
                .map(StockPriceDto::of)
                .collect(Collectors.toList());
    }

    @Override
    public void addStockPrice(StockPriceDto stockPriceDto) {

        stockPriceRepository.save(stockPriceDto.toEntity());
    }


}
