package newstock.domain.keyword.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class KeywordResponse {

    List<KeywordDto> keywords;
}
