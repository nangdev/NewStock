package newstock.domain.newsletter.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsletterResponse {

    private List<NewsletterDto> newsletterList;

    public static NewsletterResponse of(List<NewsletterDto> newsletterList) {
        return NewsletterResponse.builder()
                .newsletterList(newsletterList)
                .build();
    }
}
