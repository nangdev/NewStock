package newstock.domain.keyword.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class KeywordDto {

    String content;

    Integer stockId;

    String date;

    int count;
}
