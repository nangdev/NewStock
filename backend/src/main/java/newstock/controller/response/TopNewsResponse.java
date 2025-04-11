package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.dto.TopNewsDto;

import java.util.List;

@Getter
@Builder
public class TopNewsResponse {

    private List<TopNewsDto> newsList;

    public static TopNewsResponse of(List<TopNewsDto> topNewsDtoList) {
        return TopNewsResponse.builder()
                .newsList(topNewsDtoList)
                .build();
    }
}
