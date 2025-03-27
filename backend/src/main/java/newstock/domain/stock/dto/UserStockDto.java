package newstock.domain.stock.dto;

import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
public class UserStockDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;

    private String imgUrl;

}
