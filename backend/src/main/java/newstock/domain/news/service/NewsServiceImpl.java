package newstock.domain.news.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.request.NewsDetailRequest;
import newstock.controller.request.NewsScrapRequest;
import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.NewsDetailResponse;
import newstock.controller.response.NewsScrapResponse;
import newstock.controller.response.StockNewsResponse;
import newstock.controller.response.TopNewsResponse;
import newstock.domain.keyword.dto.Article;
import newstock.domain.news.dto.NewsDetailDto;
import newstock.domain.news.dto.NewsScrapDto;
import newstock.domain.news.dto.StockNewsDto;
import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.entity.News;
import newstock.domain.news.repository.NewsRepository;
import newstock.domain.news.repository.NewsScrapRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
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
    public TopNewsResponse getTopNewsListByStockId(Integer stockId) {

        List<News> newsList = newsRepository.getTopNewsListByStockId(stockId)
                .orElse(Collections.emptyList());

        return TopNewsResponse.of(newsList.stream()
                .map(TopNewsDto::of)
                .collect(Collectors.toList()));
    }

    @Override
    public StockNewsResponse getNewsListByStockId(StockNewsRequest stockNewsRequest) {

        Sort sort;
        Page<News> newsPage;
        Pageable pageable = PageRequest.of(stockNewsRequest.getPage(), stockNewsRequest.getCount());

        if ("score".equalsIgnoreCase(stockNewsRequest.getSort())) {
            // score 정렬의 경우 커스텀 쿼리에서 정렬하므로, pageable에 sort는 따로 추가하지 않는다.
            newsPage = newsRepository.findNewsByStockIdOrderByScoreAbs(stockNewsRequest.getStockId(), pageable);
        } else {
            // 기본 정렬: publishedDate 내림차순
            sort = Sort.by("publishedDate").descending();
            pageable = PageRequest.of(stockNewsRequest.getPage(), stockNewsRequest.getCount(), sort);
            newsPage = newsRepository.findByStockId(stockNewsRequest.getStockId(), pageable);
        }

        int totalPage = newsPage.getTotalPages();
        if(newsPage.isEmpty()){
            return StockNewsResponse.of(0, Collections.emptyList());
        }

        return StockNewsResponse.of(totalPage, newsPage.stream()
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
    public NewsScrapResponse getNewsScrapListByStockId(NewsScrapRequest newsScrapRequest) {

        Sort sort;
        if ("score".equalsIgnoreCase(newsScrapRequest.getSort())) {
            sort = Sort.by("score").descending();
        } else {
            sort = Sort.by("publishedDate").descending();
        }

        Pageable pageable = PageRequest.of(newsScrapRequest.getPage(), newsScrapRequest.getCount(), sort);

        Page<News> newsPage = newsScrapRepository.findScrappedNewsByUserIdAndStockId(
                newsScrapRequest.getUserId(),
                newsScrapRequest.getStockId(),
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

    @Transactional
    @Override
    public void deleteNewsScrapByNewsId(NewsScrapDto newsScrapDto) {

        Integer scrapId = newsScrapRepository
                .findIdByNewsIdAndUserId(newsScrapDto.getNewsId(), newsScrapDto.getUserId())
                .orElseThrow(() -> new DbException(ExceptionCode.NEWS_SCRAP_NOT_FOUND));

        newsScrapRepository.deleteById(scrapId);
    }

    @Override
    public List<Article> getNewsByStockIdAndDate(Integer stockId) {

        String today = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));

        List<News> newsList = newsRepository.findNewsByStockIdAndDate(stockId, today);

        return newsList.stream()
                .map(news -> Article.builder()
                        .content(news.getContent())
                        .build())
                .collect(Collectors.toList());
    }
}
