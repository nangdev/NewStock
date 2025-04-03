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
public class KeywordResponse {

    private List<KeywordItem> keywords;

    public static KeywordResponse of(List<KeywordItem> keywords) {
        return KeywordResponse.builder()
                .keywords(keywords)
                .build();
    }
}
