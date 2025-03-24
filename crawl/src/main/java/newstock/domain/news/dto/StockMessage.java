package newstock.domain.news.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class StockMessage {

    private String stockName;

    private String stockCode;

}
