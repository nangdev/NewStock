package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AnalysisRequest {

    private String title;

    private String content;

    public static AnalysisRequest of(String title, String content) {
        return AnalysisRequest.builder()
                .title(title)
                .content(content)
                .build();
    }
}
