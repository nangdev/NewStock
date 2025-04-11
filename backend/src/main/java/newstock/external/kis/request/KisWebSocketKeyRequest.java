package newstock.external.kis.request;

import lombok.Data;

@Data
public class KisWebSocketKeyRequest {
    private String grantType;
    private String appkey;
    private String secretkey;
}
