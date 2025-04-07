package newstock.domain.keyword.service;

import newstock.controller.request.NewsletterContentRequest;
import newstock.domain.keyword.dto.*;

public interface KeywordService {

    KeywordAIResponse extractKeywords(KeywordAIRequest request);

    void addKeyword(KeywordList keywordList);

    void addKeywordByContent(Integer stockId, NewsletterContentRequest newsletterContentRequest);

    KeywordResponse getKeywordsByStockId(KeywordRequest keywordRequest);

}
