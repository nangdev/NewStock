package newstock.domain.stock.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
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

    @Id
    private int stockCode;

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;

    private String imgUrl;

    private Integer totalPrice;

    private String capital;

    private String parValue;

    private String issuePrice;

    private String listingDate;

    private String stdIccn;

    @OneToMany(mappedBy = "stock", fetch = FetchType.LAZY)
    private List<UserStock> userStocks;

}
