package newstock.domain.stock.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.repository.StockRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class StockServiceImpl implements StockService {

    private final StockRepository stockRepository;

    @Override
    public List<StockDto> getAllStocks() {
        return stockRepository.findAllStockIdAndName();
    }
}
