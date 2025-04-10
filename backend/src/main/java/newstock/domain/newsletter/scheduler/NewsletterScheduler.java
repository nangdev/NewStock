package newstock.domain.newsletter.scheduler;

import lombok.RequiredArgsConstructor;
import newstock.domain.keyword.dto.Article;
import newstock.domain.keyword.dto.KeywordAIRequest;
import newstock.domain.keyword.dto.KeywordAIResponse;
import newstock.domain.keyword.dto.KeywordList;
import newstock.domain.keyword.service.KeywordService;
import newstock.domain.news.service.NewsService;
import newstock.domain.newsletter.service.NewsletterService;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.service.StockService;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
public class NewsletterScheduler {

    private final NewsletterService newsletterService;

    private final NewsService newsService;

    private final KeywordService keywordService;

    private final StockService stockService;

    @Scheduled(cron = "0 30 17 * * ?")
    public void scheduleNewsLetter() {

        List<StockDto> stockDtoList = stockService.getAllStockList();
        for (StockDto stockDto : stockDtoList) {

            List<Article> articles =newsService.getNewsByStockIdAndDate(stockDto.getStockId());

            if (articles.isEmpty())
                continue;

            KeywordAIResponse keywordAIResponse = keywordService.extractKeywords(KeywordAIRequest.of(articles));

            keywordService.addKeyword(KeywordList.of(keywordAIResponse.getKeywords(),stockDto.getStockId()));

            newsletterService.addNewsletter(stockDto.getStockId());
        }
    }
}
