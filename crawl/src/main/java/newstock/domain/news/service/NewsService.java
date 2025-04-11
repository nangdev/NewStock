package newstock.domain.news.service;

import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.entity.News;

import java.util.List;

public interface NewsService {

    List<NewsItem> addNewsItems(List<NewsItem> newsItemList);
}
