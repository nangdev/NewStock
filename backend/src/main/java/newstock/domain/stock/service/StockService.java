package newstock.domain.stock.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.repository.StockRepository;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class StockService {
    private final StockRepository stockRepository;

    public List<StockDto> findAll() {
        List<StockDto> stockList = stockRepository.findAll()
                .stream()
                .map(stock -> Stock.of)

    }
}
