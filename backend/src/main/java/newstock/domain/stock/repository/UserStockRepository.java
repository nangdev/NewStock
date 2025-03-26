package newstock.domain.stock.repository;

import newstock.domain.stock.entity.UserStock;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserStockRepository extends JpaRepository<UserStock, Integer> {
    void deleteUserStocksByUserId(Integer userId);
}
