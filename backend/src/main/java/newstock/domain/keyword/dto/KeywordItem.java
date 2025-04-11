package newstock.domain.keyword.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class KeywordItem {

    private String word;

    private int count;

    public static KeywordItem of(String word, int count) {
        return KeywordItem.builder()
                .word(word)
                .count(count)
                .build();
    }
}
