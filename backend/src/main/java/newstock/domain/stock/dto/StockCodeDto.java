package newstock.domain.stock.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class StockCodeDto {

    private Integer stockId;

    private String stockCode;

}
