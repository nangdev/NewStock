package newstock.external.kis.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class KisRealTimeStockPriceDto {

    private String stockCode;

    private LocalTime time;

    private int price;

    private double changeRate;

}
