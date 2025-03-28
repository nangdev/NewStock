package newstock.kafka.request;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class NewsCrawlerRequest {

    private String stockName;

    private Integer stockId;

    private String schedulerTime;

}
