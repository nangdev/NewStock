package newstock.domain.news.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class NewsItem {

    private Integer id;

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

    public void setScores(AnalysisResponse analysisResponse) {
        this.score = analysisResponse.getScore();
        this.financeScore = analysisResponse.getAspectScores().getFinance();
        this.strategyScore = analysisResponse.getAspectScores().getStrategy();
        this.governScore = analysisResponse.getAspectScores().getGovern();
        this.techScore = analysisResponse.getAspectScores().getTech();
        this.externalScore = analysisResponse.getAspectScores().getExternal();
    }
}
