package newstock.domain.stock.dto;

import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
public class UserStockDto {

    private int stockCode;

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;

    private String imgUrl;

}
