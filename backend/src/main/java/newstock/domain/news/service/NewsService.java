package newstock.domain.news.service;

import newstock.controller.request.NewsDetailRequest;
import newstock.controller.request.NewsScrapRequest;
import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.NewsDetailResponse;
import newstock.controller.response.NewsScrapResponse;
import newstock.controller.response.StockNewsResponse;
import newstock.domain.news.dto.TopNewsDto;

import java.util.List;

public interface NewsService {
    List<TopNewsDto> getTopNewsListByStockCode(int stockCode);

    StockNewsResponse getNewsListByStockCode(StockNewsRequest stockNewsRequest);

    NewsDetailResponse getNewsDetailByNewsId(NewsDetailRequest newsDetailRequest);

    NewsScrapResponse getNewsScrapListByStockCode(NewsScrapRequest newsScrapRequest);
}
