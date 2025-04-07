package newstock.domain.news.service;

import newstock.controller.request.NewsDetailRequest;
import newstock.controller.request.NewsScrapRequest;
import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.NewsDetailResponse;
import newstock.controller.response.NewsScrapResponse;
import newstock.controller.response.StockNewsResponse;
import newstock.controller.response.TopNewsResponse;
import newstock.domain.keyword.dto.Article;
import newstock.domain.keyword.dto.KeywordRequest;
import newstock.domain.news.dto.NewsScrapDto;

import java.util.List;

public interface NewsService {
    TopNewsResponse getTopNewsListByStockId(Integer stockId);

    StockNewsResponse getNewsListByStockId(StockNewsRequest stockNewsRequest);

    NewsDetailResponse getNewsDetailByNewsId(NewsDetailRequest newsDetailRequest);

    NewsScrapResponse getNewsScrapListByStockId(NewsScrapRequest newsScrapRequest);

    void addNewsScrapByNewsId(NewsScrapDto newsScrapDto);

    void deleteNewsScrapByNewsId(NewsScrapDto newsScrapDto);

    List<Article> getNewsByStockIdAndDate(Integer stockId);
}
