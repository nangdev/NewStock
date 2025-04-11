package newstock.external.kis.request;

import lombok.Data;

@Data
public class KisWebSocketConnectionRequest {
    private String trId;    // H0STCNT0로 고정
    private String trKey;   // 종목코드(6자리)
}
