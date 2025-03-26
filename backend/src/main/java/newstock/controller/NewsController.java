package newstock.controller;

import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.request.NewsDetailRequest;
import newstock.controller.request.NewsScrapRequest;
import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.NewsDetailResponse;
import newstock.controller.response.NewsScrapResponse;
import newstock.controller.response.StockNewsResponse;
import newstock.controller.response.TopNewsResponse;
import newstock.domain.news.dto.NewsScrapDto;
import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.service.NewsService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/news")
public class NewsController {

    private final NewsService newsService;

    /**
     * StockCode로 News 조회
     * @param stockCode 조회할 주식의 stockCode
     * @return 오늘 상위 뉴스 목록
     */
    @GetMapping("/top/{stockCode}")
    @Operation(summary = "stockCode로 상위 5개 뉴스 조회", description = "종목코드를 사용하여 상위 5개 뉴스를 조회합니다.")
    public Api<TopNewsResponse> getTopNewsListByStockCode(@PathVariable int stockCode) {

        List<TopNewsDto> topNewsDtoList = newsService.getTopNewsListByStockCode(stockCode);

        return Api.ok(TopNewsResponse.of(topNewsDtoList));
    }

    /**
     * StockCode로 News 조회
     * @param stockCode 조회할 주식의 stockCode
     * @return 개별 종목 뉴스
     */
    @GetMapping
    @Operation(summary = "stockCode로 개별 종목 뉴스 조회", description = "종목코드를 사용하여 개별 종목 뉴스 목록을 조회합니다.")
    public Api<StockNewsResponse> getNewsListByStockCode(
            @RequestParam(name = "stockCode") int stockCode,
            @RequestParam(name = "page") int page,
            @RequestParam(name = "count") int count,
            @RequestParam(name = "sort") String sort) {

        StockNewsResponse stockNewsResponse = newsService.getNewsListByStockCode(StockNewsRequest.of(stockCode, page, count, sort));

        return Api.ok(stockNewsResponse);
    }

    /**
     * NewsId로 News Detail 조회
     * @param newsId 조회할 주식의 newsId
     * @return 뉴스 상세 내용
     */
    @GetMapping("/v1/news/{newsId}")
    @Operation(summary = "NewsId로 뉴스 상세 내용 조회", description = "뉴스아이디를 사용하여 뉴스 상세 내용을 조회합니다.")
    public Api<NewsDetailResponse> getNewsDetailByNewsId(@PathVariable int newsId, @AuthenticationPrincipal int userId) {

        NewsDetailResponse newsDetailResponse = newsService.getNewsDetailByNewsId(NewsDetailRequest.of(newsId,userId));

        return Api.ok(newsDetailResponse);
    }

    /**
     * stockCode로 Scraped News 조회
     * @param stockCode 조회할 주식의 stockCode
     * @return 스크랩 뉴스 리스트
     */
    @GetMapping("/v1/news/scrap")
    @Operation(summary = "stockCode,userId로 스크랩 뉴스 조회", description = "종목코드와 유저 아이디를 사용하여 유저가 스크랩한 뉴스 목록을 조회합니다.")
    public Api<NewsScrapResponse> getNewsScrapListByStockCode(
            @RequestParam(name = "stockCode") int stockCode,
            @RequestParam(name = "page") int page,
            @RequestParam(name = "count") int count,
            @RequestParam(name = "sort") String sort,
            @AuthenticationPrincipal int userId) {

        NewsScrapResponse newsScrapResponse = newsService.getNewsScrapListByStockCode(NewsScrapRequest.of(stockCode,page,count,sort,userId));

        return Api.ok(newsScrapResponse);
    }

    @PostMapping("/v1/news/scrap/{newsId}")
    @Operation(summary = "newsId,userId로 뉴스 스크랩 추가", description = "뉴스아이디와 유저아이디를 통해 해당 뉴스를 스크랩합니다.")
    public Api<Void> addNewsScrapByNewsId(@PathVariable int newsId, @AuthenticationPrincipal int userId){

        newsService.addNewsScrapByNewsId(NewsScrapDto.of(userId,newsId));

        return Api.ok();
    }



}
