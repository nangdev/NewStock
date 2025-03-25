package newstock.domain.news.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class SummarizationResponse {

    @JsonProperty("summary_text")
    private String summaryText;
}
