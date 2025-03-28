package newstock.domain.stock.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.entity.Stock;
import newstock.domain.stock.repository.StockRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class StockServiceImpl implements StockService {

    private final StockRepository stockRepository;

    @Override
    public List<StockDto> getAllStocks() {
        List<Stock> stocks = stockRepository.findAll();
        return stocks.stream()
                .map(StockDto::of)
                .collect(Collectors.toList());
    }
}
