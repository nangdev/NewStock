package newstock.controller;

import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.request.NewsletterContentRequest;
import newstock.controller.request.NewsletterRequest;
import newstock.controller.response.NewsletterResponse;
import newstock.domain.keyword.service.KeywordService;
import newstock.domain.newsletter.service.NewsletterService;
import newstock.domain.stock.dto.UserStockDto;
import newstock.domain.stock.service.StockService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/newsletter")
public class NewsLetterController {

    private final NewsletterService newsletterService;

    private final KeywordService keywordService;

    private final StockService stockService;

    /**
     * date로 Newsletter 조회
     * @param date 조회할 날짜
     * @return 해당일 뉴스레터
     */
    @GetMapping("/{date}")
    @Operation(summary = "특정일 뉴스레터 조회", description = "유저의 관심종목 특정일 뉴스레터를 조회합니다.")
    public ResponseEntity<Api<NewsletterResponse>> getNewsletterByDate(@PathVariable String date, @AuthenticationPrincipal Integer userId) {
        List<UserStockDto> userStockList = stockService.getUserStockList(userId);
        NewsletterResponse newsletterResponse = newsletterService.getNewsletterByDate(NewsletterRequest.of(date,userStockList));

        return ResponseEntity.ok(Api.ok(newsletterResponse));
    }

    @PostMapping("/{stockId}")
    public ResponseEntity<Api<Void>> addNewsletter(@PathVariable Integer stockId, @RequestBody NewsletterContentRequest request) {
        newsletterService.addNewsletterByContent(stockId, request);
        return ResponseEntity.ok(Api.ok());
    }

    @PostMapping("/keywords/{stockId}")
    public ResponseEntity<Api<Void>> addKeywords(@PathVariable Integer stockId, @RequestBody NewsletterContentRequest request) {
        keywordService.addKeywordByContent(stockId, request);
        return ResponseEntity.ok(Api.ok());
    }

    @PostMapping("/{date}")
    public ResponseEntity<Api<Void>> addNewsletterAndKeywords(@PathVariable String date) {
        newsletterService.addNewsletterAndKeyword(date);
        return ResponseEntity.ok(Api.ok());
    }
}
