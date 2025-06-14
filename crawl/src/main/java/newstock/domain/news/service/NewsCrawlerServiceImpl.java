package newstock.domain.news.service;

import io.github.bonigarcia.wdm.WebDriverManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NewsItem;
import newstock.domain.news.util.CompanyKeywordUtil;
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
    private static final Duration WAIT_TIMEOUT = Duration.ofSeconds(5);
    private static final Duration OLDER_THAN_DURATION = Duration.ofMinutes(3); // 예: 스케줄러 기준 시간에서 3분 전

    // Selenium Grid URL을 환경 변수에서 가져옴
    @Value("${selenium.remote.url:http://selenium-hub:4444/wd/hub}")
    private String remoteUrl;

    /**
     * 주어진 종목명에 대한 뉴스들을 첫 페이지만 수집한 후,
     * 스케줄러 기준 시간(schedulerTime)에서 OLDER_THAN_DURATION(예: 3분 전) ~ schedulerTime 사이에 작성된 뉴스만 리스트로 반환합니다.
     */
    @Override
    public List<NewsItem> fetchNews(NewsCrawlerRequest newsCrawlerRequest) {
        WebDriver driver = null;
        try {
            driver = createWebDriver();
            WebDriverWait wait = new WebDriverWait(driver, WAIT_TIMEOUT);

            // 뉴스 목록 페이지 URL 구성 (첫 페이지만 탐색)
            String listPageUrl = "https://search.naver.com/search.naver?where=news&query="
                    + newsCrawlerRequest.getStockName() +
                    "&sm=tab_opt&sort=1&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=0" +
                    "&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall" +
                    "&is_sug_officeid=0&office_category=0&service_area=0&start=1";
            driver.get(listPageUrl);
            try {
                wait.until(ExpectedConditions.presenceOfElementLocated(By.cssSelector("ul.list_news")));
            } catch (TimeoutException e) {
                log.info("뉴스 목록을 찾을 수 없습니다. 크롤링을 종료합니다.");
                return new ArrayList<>();
            }

            List<WebElement> liElements = driver.findElements(By.cssSelector("ul.list_news li.bx"));
            if (liElements.isEmpty()) {
                return new ArrayList<>();
            }

            // 스케줄러 기준 시간과 threshold 시간(예: 3분 전) 계산
            Instant schedulerTime = Instant.parse(newsCrawlerRequest.getSchedulerTime());
            Instant thresholdTime = schedulerTime.minus(OLDER_THAN_DURATION);

            List<NewsItem> collectedNews = new ArrayList<>();
            List<NewsItem> basicNewsItems = extractBasicNewsItems(liElements);

            for (NewsItem item : basicNewsItems) {

                if (!CompanyKeywordUtil.isTitleContainsCompanyName(item.getTitle(), newsCrawlerRequest.getStockName())) {
                    continue;
                }

                enrichNewsItem(driver, wait, item);

                // content가 비어 있으면 저장하지 않습니다.
                if (isNullOrEmpty(item.getContent())) {
                    continue;
                }

                // 필수 추가 정보가 없으면 건너뜁니다.
                if (!isNewsItemValid(item)) {
                    continue;
                }

                // 작성 시간이 thresholdTime(예: 10:57:00) 이상이고 schedulerTime(예: 11:00:00) 미만인 경우에만 수집
                if (item.getPublishedDate() != null) {
                    try {
                        LocalDateTime publishedTime = LocalDateTime.parse(item.getPublishedDate(), DATE_FORMATTER);
                        Instant publishedInstant = publishedTime.atZone(ZoneId.of("Asia/Seoul")).toInstant();
                        if (publishedInstant.compareTo(thresholdTime) >= 0 && publishedInstant.isBefore(schedulerTime)) {
                            item.setStockId(newsCrawlerRequest.getStockId());
                            collectedNews.add(item);
                        }
                    } catch (Exception e) {
                        log.error("날짜 파싱 오류: {}", e.getMessage(), e);
                    }
                }
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
        options.addArguments(
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-extensions",
                "--window-size=1920,1080"
        );
        // 서버용 WebDriver 사용 예시 (주석 처리된 부분)
        // try {
        //     return new RemoteWebDriver(new URL(remoteUrl), options);
        // } catch (MalformedURLException e) {
        //     log.error("RemoteWebDriver URL 형식 오류: {}", e.getMessage(), e);
        // } catch (Exception e) {
        //     log.error("RemoteWebDriver 초기화 실패: {}", e.getMessage(), e);
        // }

        WebDriverManager.chromedriver().setup();
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

            // 본문 추출: ArticleCleaner를 사용하고, 내용이 없으면 id="dic_area"에서 직접 추출
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
        // content가 반드시 있어야 함.
        return !isNullOrEmpty(item.getContent()) &&
                !(isNullOrEmpty(item.getDescription()) &&
                        isNullOrEmpty(item.getNewsImage()) &&
                        isNullOrEmpty(item.getPress()) &&
                        isNullOrEmpty(item.getPressLogo()) &&
                        isNullOrEmpty(item.getPublishedDate()));
    }

    private boolean isNullOrEmpty(String str) {
        return str == null || str.trim().isEmpty();
    }
}
