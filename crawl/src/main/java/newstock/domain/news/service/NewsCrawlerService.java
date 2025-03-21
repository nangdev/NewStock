package newstock.domain.news.service;

public interface NewsCrawlerService {

    void fetchNews(String stockName) throws InterruptedException;
}