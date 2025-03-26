package newstock.domain.stock.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.stock.dto.UserStockDto;
import org.springframework.stereotype.Repository;

import java.util.List;

import static newstock.domain.stock.entity.QStock.stock;
import static newstock.domain.stock.entity.QUserStock.userStock;

@Repository
@RequiredArgsConstructor
public class StockRepositoryImpl implements StockRepositoryCustom {
    private final JPAQueryFactory queryFactory;

    public List<UserStockDto> findUserStocksByUserId(int userId) {
        return queryFactory
                .select(Projections.constructor(UserStockDto.class,
                        stock.stockCode,
                        stock.stockName,
                        stock.closingPrice,
                        stock.rcPdcp,
                        stock.imgUrl))
                .from(userStock)
                .join(userStock.stock, stock)
                .where(userStock.userId.eq(userId))
                .fetch();
    }
}
