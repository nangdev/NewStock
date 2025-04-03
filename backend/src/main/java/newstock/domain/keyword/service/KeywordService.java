package newstock.domain.keyword.service;

import newstock.domain.keyword.dto.*;

public interface KeywordService {

    KeywordAIResponse extractKeywords(KeywordAIRequest request);

    void addKeyword(KeywordList keywordList);

    KeywordResponse getKeywordsByStockId(KeywordRequest keywordRequest);

}
