package newstock.controller.request;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.entity.News;

@Getter
@Builder
public class NewsDetailRequest {

    private int newsId;

    private int userId;

    public static NewsDetailRequest of(int newsId, int userId) {
        return NewsDetailRequest.builder()
                .newsId(newsId)
                .userId(userId)
                .build();
    }

}
