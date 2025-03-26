package newstock.controller;

import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.request.StockNewsRequest;
import newstock.controller.response.StockNewsResponse;
import newstock.controller.response.TopNewsResponse;
import newstock.domain.news.dto.StockNewsDto;
import newstock.domain.news.dto.TopNewsDto;
import newstock.domain.news.service.NewsService;
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
    @Operation(summary = "stockCode로 개별 종목 뉴스 조회", description = "종목코드를 사용하여 개별 종목 뉴스를 조회합니다.")
    public Api<StockNewsResponse> getNewsListByStockCode(
            @RequestParam(name = "stockCode") int stockCode,
            @RequestParam(name = "page") int page,
            @RequestParam(name = "count") int count,
            @RequestParam(name = "sort") String sort) {

        StockNewsResponse stockNewsResponse = newsService.getNewsListByStockCode(StockNewsRequest.of(stockCode,page,count,sort));

        return Api.ok(stockNewsResponse);
    }



}
