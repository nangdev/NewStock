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
import newstock.domain.news.service.NewsService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/news")
public class NewsController {

    private final NewsService newsService;

    /**
     * StockCode로 News 조회
     * @param stockId 조회할 주식의 stockCode
     * @return 오늘 상위 뉴스 목록
     */
    @GetMapping("/top/{stockId}")
    @Operation(summary = "stockId 상위 5개 뉴스 조회", description = "종목코드를 사용하여 상위 5개 뉴스를 조회합니다.")
    public ResponseEntity<Api<TopNewsResponse>> getTopNewsListByStockCode(@PathVariable Integer stockId) {

        TopNewsResponse topNewsResponse = newsService.getTopNewsListByStockId(stockId);

        return ResponseEntity.ok(Api.ok(topNewsResponse));
    }

    /**
     * StockCode로 News 조회
     * @param stockId 조회할 주식의 stockCode
     * @return 개별 종목 뉴스
     */
    @GetMapping
    @Operation(summary = "stockId 개별 종목 뉴스 조회", description = "종목코드를 사용하여 개별 종목 뉴스 목록을 조회합니다.")
    public ResponseEntity<Api<StockNewsResponse>> getNewsListByStockCode(
            @RequestParam(name = "stockId") Integer stockId,
            @RequestParam(name = "page") int page,
            @RequestParam(name = "count") int count,
            @RequestParam(name = "sort") String sort) {

        StockNewsResponse stockNewsResponse = newsService.getNewsListByStockId(StockNewsRequest.of(stockId, page, count, sort));

        return ResponseEntity.ok(Api.ok(stockNewsResponse));
    }

    /**
     * NewsId로 News Detail 조회
     * @param newsId 조회할 주식의 newsId
     * @return 뉴스 상세 내용
     */
    @GetMapping("/{newsId}")
    @Operation(summary = "NewsId로 뉴스 상세 내용 조회", description = "뉴스아이디를 사용하여 뉴스 상세 내용을 조회합니다.")
    public ResponseEntity<Api<NewsDetailResponse>> getNewsDetailByNewsId(@PathVariable Integer newsId, @AuthenticationPrincipal Integer userId) {

        NewsDetailResponse newsDetailResponse = newsService.getNewsDetailByNewsId(NewsDetailRequest.of(newsId,userId));

        return ResponseEntity.ok(Api.ok(newsDetailResponse));
    }

    /**
     * stockCode로 Scraped News 조회
     * @param stockId 조회할 주식의 stockCode
     * @return 스크랩 뉴스 리스트
     */
    @GetMapping("/scrap")
    @Operation(summary = "stockId,userId로 스크랩 뉴스 조회", description = "종목코드와 유저 아이디를 사용하여 유저가 스크랩한 뉴스 목록을 조회합니다.")
    public ResponseEntity<Api<NewsScrapResponse>> getNewsScrapListByStockCode(
            @RequestParam(name = "stockId") Integer stockId,
            @RequestParam(name = "page") int page,
            @RequestParam(name = "count") int count,
            @RequestParam(name = "sort") String sort,
            @AuthenticationPrincipal Integer userId) {

        NewsScrapResponse newsScrapResponse = newsService.getNewsScrapListByStockId(NewsScrapRequest.of(stockId,page,count,sort,userId));

        return ResponseEntity.ok(Api.ok(newsScrapResponse));
    }

    /**
     * newsId Scraped News 추가
     * @param newsId 추가할 스크랩의 newsId
     */
    @PostMapping("/scrap/{newsId}")
    @Operation(summary = "newsId,userId로 뉴스 스크랩 추가", description = "뉴스아이디와 유저아이디를 통해 해당 뉴스를 스크랩합니다.")
    public ResponseEntity<Api<Void>> addNewsScrapByNewsId(@PathVariable Integer newsId, @AuthenticationPrincipal Integer userId){

        newsService.addNewsScrapByNewsId(NewsScrapDto.of(userId,newsId));

        return ResponseEntity.ok(Api.ok());
    }

    /**
     * newsId Scraped News 삭제
     * @param newsId 삭제 스크랩의 newsId
     */
    @DeleteMapping("/scrap/{newsId}")
    @Operation(summary = "newsId,userId로 뉴스 스크랩 삭제", description = "뉴스아이디와 유저아이디를 통해 해당 뉴스 스크랩을 삭제합니다.")
    public ResponseEntity<Api<Void>> deleteNewsScrapByNewsId(@PathVariable Integer newsId, @AuthenticationPrincipal Integer userId){

        newsService.deleteNewsScrapByNewsId(NewsScrapDto.of(userId,newsId));

        return ResponseEntity.ok(Api.ok());
    }



}
