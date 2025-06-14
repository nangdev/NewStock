package newstock.controller.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class EmailCheckResponse {

    @JsonProperty("isDuplicated")
    private boolean duplicated;
}