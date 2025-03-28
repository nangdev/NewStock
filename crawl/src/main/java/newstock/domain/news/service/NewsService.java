package newstock.domain.news.service;

import newstock.domain.news.dto.NewsItem;

import java.util.List;

public interface NewsService {

    void addNewsItems(List<NewsItem> newsItemList);
}
