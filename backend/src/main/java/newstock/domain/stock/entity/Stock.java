package newstock.domain.stock.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

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

}
