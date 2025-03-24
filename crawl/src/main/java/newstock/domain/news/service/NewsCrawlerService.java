package newstock.domain.news.service;

import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.dto.StockMessage;

import java.util.List;

public interface NewsCrawlerService {

    List<NewsItem> fetchNews(StockMessage stockMessage) throws InterruptedException;
}