package newstock.domain.news.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class SummarizationResponse {

    @JsonProperty("summary_content")
    private String summaryContent;
}
