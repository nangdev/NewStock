package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.dto.NewsDetailDto;

@Getter
@Builder
public class NewsDetailResponse {

    private NewsDetailDto newsInfo;

    private Boolean isScraped;

    public static NewsDetailResponse of(NewsDetailDto newsDetailDto, Boolean isScraped) {
        return NewsDetailResponse.builder()
                .newsInfo(newsDetailDto)
                .isScraped(isScraped)
                .build();
    }
}
