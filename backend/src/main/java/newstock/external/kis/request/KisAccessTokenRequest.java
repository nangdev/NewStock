package newstock.external.kis.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class KisAccessTokenRequest {
    @JsonProperty("grant_type")
    private String grantType;

    private String appkey;
    private String appsecret;
}
