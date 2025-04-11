package newstock.domain.news.service;

import newstock.domain.news.dto.NewsItem;
import newstock.kafka.request.NewsCrawlerRequest;

import java.util.List;

public interface NewsCrawlerService {

    List<NewsItem> fetchNews(NewsCrawlerRequest newsCrawlerRequest);
}