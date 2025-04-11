package newstock.domain.keyword.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class KeywordRequest {

    private String newsText;

    private Integer stockId;

    public static KeywordRequest of(String newsText, Integer stockId) {
        return KeywordRequest.builder()
                .newsText(newsText)
                .stockId(stockId)
                .build();
    }
}
