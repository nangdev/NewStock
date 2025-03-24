package newstock.external.kis;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class KisInputDto {
    @JsonProperty("tr_id")
    private String trId;

    @JsonProperty("tr_key")
    private String trKey;
}
