package newstock.domain.keyword.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class KeywordAIResponse {

    private List<KeywordItem> keywords;

    public static KeywordAIResponse of(List<KeywordItem> keywords) {
        return KeywordAIResponse.builder()
                .keywords(keywords)
                .build();
    }
}
