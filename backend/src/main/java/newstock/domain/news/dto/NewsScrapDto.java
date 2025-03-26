package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.entity.NewsScrap;

@Getter
@Builder
public class NewsScrapDto {

    private int userId;

    private int newsId;

    public static NewsScrapDto of(int userId, int newsId) {
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
