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
    private Integer id;

    private Integer userId;

    private int stockCode;

    @ManyToOne (fetch = FetchType.LAZY)
    private Stock stock;

    public static UserStock of(Integer userId, int stockCode) {
        return UserStock.builder()
                .userId(userId)
                .stockCode(stockCode)
                .build();
    }

}
