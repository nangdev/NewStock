package newstock.domain.news.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.NewsDetailRequest;
import newstock.controller.request.NewsScrapRequest;
import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.NewsDetailResponse;
import newstock.controller.response.NewsScrapResponse;
import newstock.controller.response.StockNewsResponse;
import newstock.domain.news.dto.NewsDetailDto;
import newstock.domain.news.dto.NewsScrapDto;
import newstock.domain.news.dto.StockNewsDto;
import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.entity.News;
import newstock.domain.news.repository.NewsRepository;
import newstock.domain.news.repository.NewsScrapRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import org.springframework.data.domain.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class NewsServiceImpl implements NewsService {

    private final NewsRepository newsRepository;

    private final NewsScrapRepository newsScrapRepository;

    @Override
    public List<TopNewsDto> getTopNewsListByStockCode(int stockCode) {

        List<News> newsList = newsRepository.getTopNewsListByStockCode(stockCode)
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

    @Override
    public NewsDetailResponse getNewsDetailByNewsId(NewsDetailRequest newsDetailRequest) {

        News news = newsRepository.findById(newsDetailRequest.getNewsId())
                .orElseThrow(() -> new DbException(ExceptionCode.NEWS_NOT_FOUND));

        boolean isScraped = newsScrapRepository.existsByNewsIdAndUserId(newsDetailRequest.getNewsId(), newsDetailRequest.getUserId());

        return NewsDetailResponse.of(NewsDetailDto.of(news), isScraped);
    }

    @Override
    public NewsScrapResponse getNewsScrapListByStockCode(NewsScrapRequest newsScrapRequest) {

        Sort sort;
        if ("score".equalsIgnoreCase(newsScrapRequest.getSort())) {
            sort = Sort.by("score").descending();
        } else {
            sort = Sort.by("publishedDate").descending();
        }

        Pageable pageable = PageRequest.of(newsScrapRequest.getPage(), newsScrapRequest.getCount(), sort);

        Page<News> newsPage = newsScrapRepository.findScrappedNewsByUserIdAndStockCode(
                newsScrapRequest.getUserId(),
                newsScrapRequest.getStockCode(),
                pageable
        );

        int totalPage = newsPage.getTotalPages();
        if (newsPage.isEmpty()) {
            return NewsScrapResponse.of(0, Collections.emptyList());
        }

        return NewsScrapResponse.of(totalPage,
                newsPage.stream()
                        .map(StockNewsDto::of)
                        .collect(Collectors.toList())
        );
    }

    @Transactional
    @Override
    public void addNewsScrapByNewsId(NewsScrapDto newsScrapDto) {

        newsScrapRepository.save(newsScrapDto.toEntity());
    }

    @Override
    public void deleteNewsScrapByNewsId(NewsScrapDto newsScrapDto) {

        int scrapId = newsScrapRepository
                .findIdByNewsIdAndUserId(newsScrapDto.getUserId(), newsScrapDto.getNewsId())
                .orElseThrow(() -> new DbException(ExceptionCode.NEWS_SCRAP_NOT_FOUND));

        newsScrapRepository.deleteById(scrapId);
    }

}
