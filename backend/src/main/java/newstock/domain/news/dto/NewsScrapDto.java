package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.entity.NewsScrap;

@Getter
@Builder
public class NewsScrapDto {

    private Integer userId;

    private Integer newsId;

    public static NewsScrapDto of(Integer userId, Integer newsId) {
        return NewsScrapDto.builder()
                .userId(userId)
                .newsId(newsId)
                .build();
    }

    public NewsScrap toEntity() {
        return NewsScrap.builder()
                .userId(this.getUserId())
                .newsId(this.getNewsId())
                .build();
    }

}
