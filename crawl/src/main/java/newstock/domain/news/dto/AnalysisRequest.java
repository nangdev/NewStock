package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AnalysisRequest {

    private String content;

    public static AnalysisRequest of(String content) {
        return AnalysisRequest.builder()
                .content(content)
                .build();
    }
}
