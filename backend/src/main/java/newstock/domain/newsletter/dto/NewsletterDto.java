package newstock.domain.newsletter.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.keyword.dto.KeywordItem;
import newstock.domain.newsletter.entity.Newsletter;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsletterDto {

    private Integer stockId;

    private String content;

    private List<KeywordItem> keywordList;

    public static NewsletterDto of(Integer stockId, String content, List<KeywordItem> keywordList) {
        return NewsletterDto.builder()
                .stockId(stockId)
                .content(content)
                .keywordList(keywordList)
                .build();
    }
}
