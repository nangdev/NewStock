package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class SummarizationRequest {

    private String content;

    private int maxLength;

    private int minLength;

    private boolean doSample;

    public static SummarizationRequest of(String content, int maxLength, int minLength, boolean doSample) {
        return SummarizationRequest.builder()
                .content(content)
                .maxLength(maxLength)
                .minLength(minLength)
                .doSample(doSample)
                .build();
    }
}
