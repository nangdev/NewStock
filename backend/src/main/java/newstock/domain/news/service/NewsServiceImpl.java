package newstock.domain.news.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.StockNewsResponse;
import newstock.domain.news.dto.StockNewsDto;
import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.entity.News;
import newstock.domain.news.repository.NewsCustomRepository;
import newstock.domain.news.repository.NewsRepository;
import org.springframework.data.domain.*;
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
    public List<TopNewsDto> getTopNewsListByStockCode(int stockCode) {
        List<News> newsList = newsCustomRepository.getTopNewsListByStockCode(stockCode)
                .orElse(Collections.emptyList());

        return newsList.stream()
                .map(TopNewsDto::of)
                .collect(Collectors.toList());
    }

    @Override
    public StockNewsResponse getNewsListByStockCode(StockNewsRequest stockNewsRequest) {
        Sort sort;
        if ("score".equalsIgnoreCase(stockNewsRequest.getSort())) {
            sort = Sort.by("score").descending();
        } else {
            sort = Sort.by("publishedDate").descending();
        }

        Pageable pageable = PageRequest.of(stockNewsRequest.getPage(), stockNewsRequest.getCount(), sort);

        Page<News> newsPage = newsRepository.findByStockCode(stockNewsRequest.getStockCode(), pageable);

        int totalPage = newsPage.getTotalPages();
        if(newsPage.isEmpty()){
            return StockNewsResponse.of(0, Collections.emptyList());
        }

        return StockNewsResponse.of(totalPage,newsPage.stream()
                .map(StockNewsDto::of)
                .collect(Collectors.toList()));
    }


}
