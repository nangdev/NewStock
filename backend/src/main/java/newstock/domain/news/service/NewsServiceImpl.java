package newstock.domain.news.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.entity.News;
import newstock.domain.news.repository.NewsCustomRepository;
import newstock.domain.news.repository.NewsRepository;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class NewsServiceImpl implements NewsService {

    private final NewsRepository newsRepository;

    private final NewsCustomRepository newsCustomRepository;

    @Override
    public List<TopNewsDto> getTopNewsByStockCode(int stockCode) {
        List<News> newsList = newsCustomRepository.getTopNewsByStockCode(stockCode)
                .orElse(Collections.emptyList());

        return newsList.stream()
                .map(TopNewsDto::of)
                .collect(Collectors.toList());
    }

}
