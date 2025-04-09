package newstock.domain.news.dto;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.news.entity.News;

@Getter
@Builder
public class NewsDetailDto {

    private String title;

    private String content;

    private String newsImage;

    private String url;

    private String press;

    private String pressLogo;

    private String publishedDate;

    private String newsSummary;

    private float score;

    private float financeScore;

    private float strategyScore;

    private float governScore;

    private float techScore;

    private float externalScore;


    public static NewsDetailDto of(News news) {
        return NewsDetailDto.builder()
                .title(news.getTitle())
                .content(news.getContent())
                .newsImage(news.getNewsImage())
                .url(news.getUrl())
                .press(news.getPress())
                .pressLogo(news.getPressLogo())
                .publishedDate(news.getPublishedDate())
                .newsSummary(news.getNewsSummary())
                .score(news.getScore())
                .financeScore(news.getFinanceScore())
                .strategyScore(news.getStrategyScore())
                .governScore(news.getGovernScore())
                .techScore(news.getTechScore())
                .externalScore(news.getExternalScore())
                .build();
    }
}
