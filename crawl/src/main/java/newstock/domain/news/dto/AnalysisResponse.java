package newstock.domain.news.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AnalysisResponse {

    private String content;

    private float score;

    @JsonProperty("aspect_scores")
    private AspectScores aspectScores;

    @Data
    @Builder
    public static class AspectScores{

        private float finance;

        private float strategy;

        private float govern;

        private float tech;

        private float external;
    }
}
