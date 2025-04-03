package newstock.domain.newsletter.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.domain.stock.dto.UserStockDto;

import java.util.List;
import java.util.stream.Collectors;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsletterRequest {

    private String date;

    private List<Integer> stockIdList;

    public static NewsletterRequest of(String date, List<UserStockDto> userStockDtos) {
        return NewsletterRequest.builder()
                .date(date)
                .stockIdList(userStockDtos.stream().map(UserStockDto::getStockId).collect(Collectors.toList()))
                .build();
    }
}
