package newstock.domain.stock.repository;

import newstock.domain.stock.entity.UserStock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;


@Repository
public interface UserStockRepository extends JpaRepository<UserStock, Integer> , UserStockRepositoryCustom{
}
