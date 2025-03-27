package newstock.domain.stock.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.dto.StockInfoDto;
import newstock.domain.stock.dto.UserStockDto;
import newstock.domain.stock.entity.Stock;
import newstock.domain.stock.entity.UserStock;
import newstock.domain.stock.repository.StockRepository;
import newstock.domain.stock.repository.UserStockRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class StockService {

    private final UserStockRepository userStockRepository;

    private final StockRepository stockRepository;

    public List<StockDto> getStockList() {
        return stockRepository.findAll()
                .stream()
                .map(StockDto::of)
                .toList();
    }

    public List<UserStockDto> getUserStockList(int userId) {
        return stockRepository.findUserStocksByUserId(userId);
    }

    @Transactional
    public void updateUserStockList(int userId, List<Integer> stockCodeList){
        try {
            userStockRepository.deleteUserStocksByUserId(userId);

            for (Integer stockCode : stockCodeList) {
                userStockRepository.save(UserStock.of(userId, stockCode));
            }

        }catch (Exception e){
            throw new DbException(ExceptionCode.USER_STOCK_UPDATE_FAILED);
        }
    }

    public StockInfoDto getStockInfo(int stockCode){
        Stock stock = stockRepository.findById(stockCode)
                .orElseThrow(() -> new DbException(ExceptionCode.STOCK_NOT_FOUND));

        return StockInfoDto.of(stock);
    }

}
