package newstock.domain.keyword.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class KeywordRequest {

    private Integer stockId;

    private String date;

    public static KeywordRequest of(Integer stockId, String date) {
        return KeywordRequest.builder()
                .stockId(stockId)
                .date(date)
                .build();
    }
}
