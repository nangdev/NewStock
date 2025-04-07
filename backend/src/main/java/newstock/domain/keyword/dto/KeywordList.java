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
public class KeywordList {

    private List<KeywordItem> keywords;

    private Integer stockId;

    public static KeywordList of(List<KeywordItem> keywords, Integer stockId) {
        return KeywordList.builder()
                .keywords(keywords)
                .stockId(stockId)
                .build();
    }
}
