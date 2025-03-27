package newstock.domain.stock.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class Stock {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer stockId;

    private String stockCode;

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;

    private String imgUrl;

    private Integer totalPrice;

    private String capital;

    private String lstgStqt;

    private String listingDate;

    private String stdIccn;

    @OneToMany(mappedBy = "stock", fetch = FetchType.LAZY)
    private List<UserStock> userStocks;

}
