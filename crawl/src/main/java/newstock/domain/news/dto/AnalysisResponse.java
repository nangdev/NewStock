package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AnalysisResponse {

    private String content;

    private int score;
}
