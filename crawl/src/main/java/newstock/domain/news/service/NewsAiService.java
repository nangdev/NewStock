package newstock.domain.news.service;

import newstock.domain.keyword.dto.KeywordRequest;
import newstock.domain.news.dto.AnalysisRequest;
import newstock.domain.news.dto.AnalysisResponse;
import newstock.domain.keyword.dto.KeywordResponse;
import newstock.domain.news.dto.SummarizationRequest;
import newstock.domain.news.dto.SummarizationResponse;

public interface NewsAiService {

    SummarizationResponse summarize(SummarizationRequest summarizationRequest);

    AnalysisResponse analysis(AnalysisRequest analysisRequest);

    KeywordResponse getKeyword(KeywordRequest keywordRequest);
}
