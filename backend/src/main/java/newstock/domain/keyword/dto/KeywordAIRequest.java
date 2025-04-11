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
public class KeywordAIRequest {

    private List<Article> articles;

    public static KeywordAIRequest of(List<Article> articles) {
        return KeywordAIRequest.builder()
                .articles(articles)
                .build();
    }
}
