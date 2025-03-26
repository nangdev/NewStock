package newstock.domain.news.service;

import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.StockNewsResponse;
import newstock.domain.news.dto.StockNewsDto;
import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.entity.News;

import java.util.List;

public interface NewsService {
    List<TopNewsDto> getTopNewsListByStockCode(int stockCode);

    StockNewsResponse getNewsListByStockCode(StockNewsRequest stockNewsRequest);
}
