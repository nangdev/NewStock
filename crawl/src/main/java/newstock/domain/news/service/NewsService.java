package newstock.domain.news.service;

import newstock.domain.news.dto.NewsItem;

import java.util.List;

public interface NewsService {

    void saveNewsItems(List<NewsItem> newsItemList);
}
