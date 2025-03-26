package newstock.domain.news.service;

import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.entity.News;

import java.util.List;

public interface NewsService {
    List<TopNewsDto> getTopNewsByStockCode(int stockCode);
}
