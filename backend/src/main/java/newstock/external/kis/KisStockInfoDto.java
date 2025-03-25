package newstock.external.kis;

import lombok.*;

import java.time.LocalTime;

@Builder
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class KisStockInfoDto {
    private String stockCode;
    private LocalTime time;
    private int price;
    private double changeRate;
}
