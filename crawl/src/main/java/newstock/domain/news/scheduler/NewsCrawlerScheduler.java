package newstock.domain.news.scheduler;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.service.NewsCrawlerService;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class NewsCrawlerScheduler {

    private final NewsCrawlerService newsCrawlerService;

    // 매 분 0초마다 실행 (cron expression: "0 * * * * *")
    @Scheduled(cron = "0 * * * * *")
    public void scheduleNewsCrawling() {
        // 크롤링할 종목 리스트 정의
        List<String> stockNames = List.of("삼성전자", "현대자동차", "LG전자", "네이버", "카카오");

        for (String stock : stockNames) {
            try {
                newsCrawlerService.fetchNews(stock);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.error("뉴스 크롤링 인터럽트 발생 ({}): {}", stock, e.getMessage());
            } catch (Exception e) {
                log.error("뉴스 크롤링 중 오류 발생 ({}): {}", stock, e.getMessage());
            }
        }
    }
}
