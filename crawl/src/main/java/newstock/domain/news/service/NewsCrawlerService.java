package newstock.domain.news.service;

import newstock.domain.news.dto.NewsItem;

import java.util.List;

public interface NewsCrawlerService {

    List<NewsItem> fetchNews(String stockName) throws InterruptedException;
}