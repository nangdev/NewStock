package newstock.domain.news.service;

import newstock.domain.keyword.dto.KeywordRequest;
import newstock.domain.news.dto.AnalysisResponse;
import newstock.domain.keyword.dto.KeywordResponse;
import newstock.domain.news.dto.SummarizationResponse;

public interface NewsAiService {

    SummarizationResponse summarize(String content, int maxLength, int minLength, boolean doSample);

    AnalysisResponse analysis(String newsText);

    KeywordResponse getKeyword(KeywordRequest keywordRequest);
}
