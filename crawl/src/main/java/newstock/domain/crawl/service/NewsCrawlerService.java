package newstock.domain.crawl.service;

public interface NewsCrawlerService {

    void fetchNews(String stockName) throws InterruptedException;
}