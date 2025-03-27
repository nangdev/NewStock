package newstock.domain.stock.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserStock {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    private int userId;

    private int stockCode;

    @ManyToOne (fetch = FetchType.LAZY)
    private Stock stock;

    public static UserStock of(int userId, int stockCode) {
        return UserStock.builder()
                .userId(userId)
                .stockCode(stockCode)
                .build();
    }

}
