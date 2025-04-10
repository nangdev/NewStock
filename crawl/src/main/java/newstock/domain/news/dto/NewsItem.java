package newstock.domain.news.dto;

import lombok.*;
import newstock.domain.news.entity.News;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsItem {

    private Integer newsId;

    private Integer stockId;

    private String title;

    private String description;

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

    public static NewsItem of(News news){
        return NewsItem.builder()
                .newsId(news.getNewsId())
                .stockId(news.getNewsId())
                .title(news.getTitle())
                .description(news.getDescription())
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

    public void setScores(AnalysisResponse analysisResponse) {
        this.score = analysisResponse.getScore();
        this.financeScore = analysisResponse.getAspectScores().getFinance();
        this.strategyScore = analysisResponse.getAspectScores().getStrategy();
        this.governScore = analysisResponse.getAspectScores().getGovern();
        this.techScore = analysisResponse.getAspectScores().getTech();
        this.externalScore = analysisResponse.getAspectScores().getExternal();
    }
}
