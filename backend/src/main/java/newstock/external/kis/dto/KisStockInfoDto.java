package newstock.external.kis.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Builder
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class KisStockInfoDto {

    private String closingPrice; // 당일 종가 (stck_prpr)

    private String ctpdPrice;   // 전일 대비 등락 가격 (prdy_vrss)

    private String rcPdcp;      // 전일 대비 등락률 (prdy_ctrt)

    private String totalPrice;  // HTS 시가 총액 (hts_avls)

}
