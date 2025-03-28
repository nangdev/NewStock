package newstock.kafka.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsCrawlerRequest {

    private String stockName;

    private Integer stockId;

    private String schedulerTime;

    public static NewsCrawlerRequest of(String stockName, Integer stockId, String schedulerTime) {
        return NewsCrawlerRequest.builder()
                .stockName(stockName)
                .stockId(stockId)
                .schedulerTime(schedulerTime)
                .build();
    }
}
