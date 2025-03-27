package newstock.domain.news.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.entity.News;
import newstock.domain.news.repository.NewsRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class NewsServiceImpl implements NewsService {

    private final NewsRepository newsRepository;

    @Transactional
    @Override
    public void addNewsItems(List<NewsItem> newsItemList) {

        List<News> newsEntities = newsItemList.stream()
                .map(News::of)
                .collect(Collectors.toList());

        newsRepository.saveAll(newsEntities);
    }
}
