package newstock.domain.stock.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Builder
@NoArgsConstructor
@AllArgsConstructor
@Getter
public class UserStockDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;

    private String imgUrl;

}
