package newstock.controller;

import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.domain.keyword.dto.KeywordResponse;
import newstock.domain.newsletter.dto.NewsletterRequest;
import newstock.domain.newsletter.dto.NewsletterResponse;
import newstock.domain.newsletter.service.NewsletterService;
import newstock.domain.stock.dto.UserStockDto;
import newstock.domain.stock.service.StockService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/newsletter")
public class NewsLetterController {

    private final NewsletterService newsletterService;

    private final StockService stockService;

    /**
     * date로 Newsletter 조회
     * @param date 조회할 날짜
     * @return 해당일 뉴스레터
     */
    @GetMapping("/{date}")
    @Operation(summary = "특정일 뉴스레터 조회", description = "유저의 관심종목 특정일 뉴스레터를 조회합니다.")
    public ResponseEntity<Api<NewsletterResponse>> getNewsLetterByDate(@PathVariable String date, @AuthenticationPrincipal Integer userId) {
        List<UserStockDto> userStockList = stockService.getUserStockList(userId);
        NewsletterResponse newsletterResponse = newsletterService.getNewsletterByDate(NewsletterRequest.of(date,userStockList));

        return ResponseEntity.ok(Api.ok(newsletterResponse));
    }

}
