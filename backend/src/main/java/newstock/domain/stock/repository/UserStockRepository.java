package newstock.domain.stock.repository;

import newstock.domain.stock.entity.UserStock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface UserStockRepository extends JpaRepository<UserStock, Integer> {

    @Modifying
    @Query("delete from UserStock us where us.userId = :userId")
    void deleteUserStocksByUserId(Integer userId);

    @Query("select us from UserStock us where us.userId = :userId")
    List<UserStock> findUserStocksByUserId(Integer userId);

}
