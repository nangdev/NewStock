package newstock.domain.stock.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.querydsl.core.types.Projections;
import jakarta.persistence.EntityManager;
import lombok.RequiredArgsConstructor;
import newstock.domain.stock.dto.StockDto;
import org.springframework.stereotype.Repository;

import static newstock.domain.stock.entity.QStock.stock;
import java.util.List;

@RequiredArgsConstructor
@Repository
public class StockCustomRepositoryImpl implements StockCustomRepository {

    private final JPAQueryFactory queryFactory;

    @Override
    public List<StockDto> findAllStockIdAndName() {
        return queryFactory
                .select(Projections.constructor(StockDto.class,
                        stock.stockId,
                        stock.stockName))
                .from(stock)
                .fetch();
    }
}
