package newstock.domain.news.service;

import io.github.bonigarcia.wdm.WebDriverManager;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.dto.StockMessage;
import newstock.domain.news.util.ArticleCleaner;
import org.openqa.selenium.By;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
public class NewsCrawlerServiceImpl implements NewsCrawlerService {

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final Duration WAIT_TIMEOUT = Duration.ofSeconds(10);
    private static final Duration OLDER_THAN_DURATION = Duration.ofMinutes(1); // 테스트용 1분 전 기준

    /**
     * 주어진 종목명에 대한 뉴스들을 수집한 후 리스트로 반환합니다.
     * 나중에 AI 필터링 등의 추가 처리를 위해 이 리스트를 활용할 수 있습니다.
     */
    @Override
    public List<NewsItem> fetchNews(StockMessage stockMessage) throws InterruptedException {
        // WebDriver 설정
        WebDriverManager.chromedriver().browserVersion("134.0.6998.89").setup();
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--headless");
        WebDriver driver = new ChromeDriver(options);
        WebDriverWait wait = new WebDriverWait(driver, WAIT_TIMEOUT);

        // 뉴스 목록 페이지 URL 구성
        String baseUrl = "https://search.naver.com/search.naver?where=news&query=" + stockMessage.getStockName() +
                "&sm=tab_opt&sort=1&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=0" +
                "&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall" +
                "&is_sug_officeid=0&office_category=0&service_area=0";
        int start = 1;
        boolean isLastPage = false;
        boolean stopCrawling = false;
        // 임계 시간: 현재 시간 기준 OLDER_THAN_DURATION 이전인 뉴스는 수집하지 않음
        Instant thresholdTime = Instant.now().minus(OLDER_THAN_DURATION);

        // 수집된 뉴스 항목들을 저장할 리스트
        List<NewsItem> collectedNews = new ArrayList<>();

        // 목록 페이지를 순회하며 뉴스 수집
        while (!isLastPage && !stopCrawling) {
            String listPageUrl = baseUrl + "&start=" + start;
            driver.get(listPageUrl);

            try {
                wait.until(ExpectedConditions.presenceOfElementLocated(By.cssSelector("ul.list_news")));
            } catch (TimeoutException e) {
                log.info("뉴스 목록을 찾을 수 없습니다. 크롤링을 종료합니다.");
                break;
            }

            List<WebElement> liElements = driver.findElements(By.cssSelector("ul.list_news li.bx"));
            if (liElements.isEmpty()) {
                isLastPage = true;
                break;
            }

            // 1단계: 목록 페이지에서 기본 정보(제목, 네이버뉴스 링크) 추출
            List<NewsItem> basicNewsItems = extractBasicNewsItems(liElements);

            // 2단계: 각 뉴스 항목별로 개별 기사 페이지에서 추가 정보 수집
            for (NewsItem item : basicNewsItems) {
                enrichNewsItem(driver, wait, item);
                log.info("수집 뉴스: 제목: {}, 작성시간: {}", item.getTitle(), item.getPublishedDate());

                // 필수 추가 정보가 없으면 건너뜁니다.
                if (!isNewsItemValid(item)) {
                    continue;
                }

                if (item.getPublishedDate() != null) {
                    LocalDateTime publishedTime = LocalDateTime.parse(item.getPublishedDate(), DATE_FORMATTER);
                    Instant publishedInstant = publishedTime.atZone(ZoneId.of("Asia/Seoul")).toInstant();
                    if (publishedInstant.isBefore(thresholdTime)) {
                        stopCrawling = true;
                        break;
                    }
                }
                item.setStockCode(stockMessage.getStockCode());
                collectedNews.add(item);
            }

            if (stopCrawling) {
                break;
            }
            start += 10;
            int sleepTime = 1000 + (int) (Math.random() * 1000);
            Thread.sleep(sleepTime);
        }
        driver.quit();

        for (NewsItem news : collectedNews) {
            log.info(news.toString());
        }
        return collectedNews;
    }

    /**
     * 목록 페이지에서 li 태그들을 순회하며 기본 정보(제목, 네이버뉴스 링크)를 추출하여 리스트로 반환합니다.
     */
    private List<NewsItem> extractBasicNewsItems(List<WebElement> liElements) {
        List<NewsItem> basicNewsItems = new ArrayList<>();
        for (WebElement li : liElements) {
            try {
                List<WebElement> infoLinkElems = li.findElements(
                        By.xpath(".//div[contains(@class, 'info_group')]//a[contains(text(), '네이버뉴스')]")
                );
                if (infoLinkElems.isEmpty()) {
                    continue;
                }
                String newsLink = infoLinkElems.get(0).getAttribute("href");
                if (newsLink == null || !newsLink.contains("n.news.naver.com")) {
                    continue;
                }
                WebElement titleElem = li.findElement(By.cssSelector("a.news_tit"));
                if (titleElem == null) {
                    continue;
                }
                String title = titleElem.getAttribute("title");
                if (title == null || title.trim().isEmpty()) {
                    title = titleElem.getText();
                }
                NewsItem newsItem = new NewsItem();
                newsItem.setTitle(title);
                newsItem.setUrl(newsLink);
                basicNewsItems.add(newsItem);
            } catch (Exception ex) {
                log.error("목록 페이지에서 뉴스 항목 기본 정보 추출 중 오류 발생: {}", ex.getMessage());
            }
        }
        return basicNewsItems;
    }

    /**
     * 개별 뉴스 기사 페이지로 이동하여 추가 정보를 수집합니다.
     */
    private void enrichNewsItem(WebDriver driver, WebDriverWait wait, NewsItem item) {
        try {
            driver.get(item.getUrl());
            wait.until(ExpectedConditions.presenceOfElementLocated(By.tagName("body")));
            String articleHtml = driver.getPageSource();
            var doc = org.jsoup.Jsoup.parse(articleHtml);

            // 본문 추출: ArticleCleaner를 사용하고, 내용이 없으면 <article id="dic_area">에서 직접 추출
            String content = ArticleCleaner.extractMeaningfulContent(articleHtml);
            if (content == null || content.trim().isEmpty()) {
                var articleElem = doc.getElementById("dic_area");
                if (articleElem != null) {
                    content = articleElem.text().trim();
                }
            }
            item.setContent(content);

            // 메타 태그에서 설명과 대표 이미지 추출
            var metaDesc = doc.selectFirst("meta[property=og:description]");
            if (metaDesc != null) {
                item.setDescription(metaDesc.attr("content"));
            }
            var metaImage = doc.selectFirst("meta[property=og:image]");
            if (metaImage != null) {
                item.setNewsImage(metaImage.attr("content"));
            }

            // 언론사 로고 및 이름 추출
            List<WebElement> logoElems = driver.findElements(By.cssSelector("img.media_end_head_top_logo_img.light_type"));
            if (!logoElems.isEmpty()) {
                WebElement logoElem = logoElems.get(0);
                String pressLogo = logoElem.getAttribute("data-src");
                if (pressLogo == null || pressLogo.trim().isEmpty()) {
                    pressLogo = logoElem.getAttribute("src");
                }
                item.setPressLogo(pressLogo);
                String pressName = logoElem.getAttribute("alt");
                if (pressName == null || pressName.trim().isEmpty()) {
                    pressName = logoElem.getAttribute("title");
                }
                item.setPress(pressName);
            }

            // 작성 시간 추출
            List<WebElement> timeElems = driver.findElements(
                    By.cssSelector("span.media_end_head_info_datestamp_time._ARTICLE_DATE_TIME")
            );
            if (!timeElems.isEmpty()) {
                String publishedStr = timeElems.get(0).getAttribute("data-date-time");
                item.setPublishedDate(publishedStr);
            }
        } catch (Exception ex) {
            log.error("기사 페이지 파싱 중 오류 발생 ({}): {}", item.getUrl(), ex.getMessage());
            item.setContent("");
        }
    }

    /**
     * 필수 추가 정보(본문, 설명, 대표 이미지, 언론사, 로고, 작성시간)가 모두 존재하는지 검증합니다.
     */
    private boolean isNewsItemValid(NewsItem item) {
        return !((item.getContent() == null || item.getContent().trim().isEmpty()) &&
                (item.getDescription() == null || item.getDescription().trim().isEmpty()) &&
                (item.getNewsImage() == null || item.getNewsImage().trim().isEmpty()) &&
                (item.getPress() == null || item.getPress().trim().isEmpty()) &&
                (item.getPressLogo() == null || item.getPressLogo().trim().isEmpty()) &&
                (item.getPublishedDate() == null || item.getPublishedDate().trim().isEmpty()));
    }
}
