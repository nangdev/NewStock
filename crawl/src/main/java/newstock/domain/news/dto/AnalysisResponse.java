package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AnalysisResponse {

    private String content;

    private float score;

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
