package newstock.external.kis.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class KisWebSocketKeyResponse {
    @JsonProperty("approval_key")
    private String approvalKey;
}
