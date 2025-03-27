package newstock.domain.stock.repository;

import newstock.domain.stock.dto.UserStockDto;

import java.util.List;

public interface StockRepositoryCustom {

    List<UserStockDto> findUserStocksByUserId(int userId);

}
