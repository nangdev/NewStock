package newstock.domain.news.service;

import io.github.bonigarcia.wdm.WebDriverManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NewsItem;
import newstock.kafka.request.NewsCrawlerRequest;
import newstock.domain.news.util.ArticleCleaner;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.openqa.selenium.By;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.net.MalformedURLException;
import java.net.URL;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
@RequiredArgsConstructor
public class NewsCrawlerServiceImpl implements NewsCrawlerService {

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final Duration WAIT_TIMEOUT = Duration.ofSeconds(3);
    private static final Duration OLDER_THAN_DURATION = Duration.ofMinutes(1); // 테스트용 1분 전 기준
    private static final int PAGE_INCREMENT = 10;

    // Selenium Grid URL을 환경 변수에서 가져옴
    @Value("${selenium.remote.url:http://selenium-hub:4444/wd/hub}")
    private String remoteUrl;

    /**
     * 주어진 종목명에 대한 뉴스들을 수집한 후 리스트로 반환합니다.
     * 나중에 AI 필터링 등의 추가 처리를 위해 이 리스트를 활용할 수 있습니다.
     */
    @Override
    public List<NewsItem> fetchNews(NewsCrawlerRequest newsCrawlerRequest) throws InterruptedException {
        WebDriver driver = null;
        try {
            driver = createWebDriver();
            WebDriverWait wait = new WebDriverWait(driver, WAIT_TIMEOUT);

            // 뉴스 목록 페이지 URL 구성
            String baseUrl = "https://search.naver.com/search.naver?where=news&query="
                    + newsCrawlerRequest.getStockName() +
                    "&sm=tab_opt&sort=1&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=0" +
                    "&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall" +
                    "&is_sug_officeid=0&office_category=0&service_area=0";
            int start = 1;
            boolean isLastPage = false;
            boolean stopCrawling = false;

            // 스케줄러에서 전달받은 시간을 기준으로 파싱
            Instant schedulerTime = Instant.parse(newsCrawlerRequest.getSchedulerTime());
            // OLDER_THAN_DURATION 이전의 뉴스는 수집하지 않도록 기준 시간을 계산
            Instant thresholdTime = schedulerTime.minus(OLDER_THAN_DURATION);

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

                List<NewsItem> basicNewsItems = extractBasicNewsItems(liElements);

                for (NewsItem item : basicNewsItems) {
                    enrichNewsItem(driver, wait, item);

                    // 필수 추가 정보가 없으면 건너뜁니다.
                    if (!isNewsItemValid(item)) {
                        continue;
                    }

                    if (item.getPublishedDate() != null) {
                        try {
                            LocalDateTime publishedTime = LocalDateTime.parse(item.getPublishedDate(), DATE_FORMATTER);
                            Instant publishedInstant = publishedTime.atZone(ZoneId.of("Asia/Seoul")).toInstant();
                            if (publishedInstant.isBefore(thresholdTime)) {
                                stopCrawling = true;
                                break;
                            }
                        } catch (Exception e) {
                            log.error("날짜 파싱 오류: {}", e.getMessage(), e);
                        }
                    }
                    item.setStockId(newsCrawlerRequest.getStockId());
                    collectedNews.add(item);
                }

                if (stopCrawling) {
                    break;
                }
                start += PAGE_INCREMENT;
                // 500ms ~ 1000ms 사이의 딜레이
                Thread.sleep(500 + (int)(Math.random() * 500));
            }
            return collectedNews;
        } catch (Exception e) {
            log.error("뉴스 크롤링 중 오류 발생: {}", e.getMessage(), e);
            return new ArrayList<>();
        } finally {
            if (driver != null) {
                try {
                    driver.quit();
                } catch (Exception e) {
                    log.error("WebDriver 종료 중 오류 발생: {}", e.getMessage(), e);
                }
            }
        }
    }

    private WebDriver createWebDriver() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--headless", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu");

        // 원격 WebDriver 시도
        try {
            log.info("RemoteWebDriver 초기화 시작. URL: {}", remoteUrl);
            WebDriver driver = new RemoteWebDriver(new URL(remoteUrl), options);
            log.info("RemoteWebDriver 초기화 성공");
            return driver;
        } catch (MalformedURLException e) {
            log.error("RemoteWebDriver URL 형식 오류: {}", e.getMessage(), e);
        } catch (Exception e) {
            log.error("RemoteWebDriver 초기화 실패: {}", e.getMessage(), e);
        }

        // 자동 폴백 (로컬 WebDriver 사용)
        log.info("로컬 WebDriver로 폴백합니다.");
        WebDriverManager.chromedriver().browserVersion("134.0.6998.89").setup();
        return new ChromeDriver(options);
    }

    /**
     * 목록 페이지에서 li 태그들을 순회하며 기본 정보(제목, 네이버뉴스 링크)를 추출하여 리스트로 반환합니다.
     */
    private List<NewsItem> extractBasicNewsItems(List<WebElement> liElements) {
        List<NewsItem> basicNewsItems = new ArrayList<>();
        for (WebElement li : liElements) {
            try {
                List<WebElement> infoLinkElems = li.findElements(By.xpath(".//div[contains(@class, 'info_group')]//a[contains(text(), '네이버뉴스')]"));
                if (infoLinkElems.isEmpty()) {
                    continue;
                }
                String newsLink = infoLinkElems.get(0).getAttribute("href");
                if (newsLink == null || !newsLink.contains("n.news.naver.com")) {
                    continue;
                }
                WebElement titleElem = li.findElement(By.cssSelector("a.news_tit"));
                String title = titleElem.getAttribute("title");
                if (title == null || title.trim().isEmpty()) {
                    title = titleElem.getText();
                }
                NewsItem newsItem = new NewsItem();
                newsItem.setTitle(title);
                newsItem.setUrl(newsLink);
                basicNewsItems.add(newsItem);
            } catch (Exception ex) {
                log.error("목록 페이지에서 뉴스 항목 기본 정보 추출 중 오류 발생: {}", ex.getMessage(), ex);
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
            Document doc = Jsoup.parse(articleHtml);

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
            List<WebElement> timeElems = driver.findElements(By.cssSelector("span.media_end_head_info_datestamp_time._ARTICLE_DATE_TIME"));
            if (!timeElems.isEmpty()) {
                String publishedStr = timeElems.get(0).getAttribute("data-date-time");
                item.setPublishedDate(publishedStr);
            }
        } catch (Exception ex) {
            log.error("기사 페이지 파싱 중 오류 발생 ({}): {}", item.getUrl(), ex.getMessage(), ex);
            item.setContent("");
        }
    }

    /**
     * 필수 추가 정보(본문, 설명, 대표 이미지, 언론사, 로고, 작성시간)가 모두 존재하는지 검증합니다.
     */
    private boolean isNewsItemValid(NewsItem item) {
        return !(isNullOrEmpty(item.getContent()) &&
                isNullOrEmpty(item.getDescription()) &&
                isNullOrEmpty(item.getNewsImage()) &&
                isNullOrEmpty(item.getPress()) &&
                isNullOrEmpty(item.getPressLogo()) &&
                isNullOrEmpty(item.getPublishedDate()));
    }

    private boolean isNullOrEmpty(String str) {
        return str == null || str.trim().isEmpty();
    }
}
