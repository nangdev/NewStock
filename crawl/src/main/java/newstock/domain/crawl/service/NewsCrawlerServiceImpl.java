package newstock.domain.crawl.service;

import io.github.bonigarcia.wdm.WebDriverManager;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.crawl.dto.NewsItem;
import newstock.domain.crawl.util.ArticleCleaner;
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

    @Override
    public void fetchNews(String stockName) throws InterruptedException {
        WebDriverManager.chromedriver().browserVersion("134.0.6998.89").setup();
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--headless");
        WebDriver driver = new ChromeDriver(options);
        WebDriverWait wait = new WebDriverWait(driver, WAIT_TIMEOUT);

        String baseUrl = "https://search.naver.com/search.naver?where=news&query=" + stockName +
                "&sm=tab_opt&sort=1&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=0" +
                "&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall" +
                "&is_sug_officeid=0&office_category=0&service_area=0";
        int start = 1;
        boolean isLastPage = false;
        boolean stopCrawling = false;
        Instant thresholdTime = Instant.now().minus(OLDER_THAN_DURATION);

        List<NewsItem> collectedNews = new ArrayList<>();

        while (!isLastPage && !stopCrawling) {
            String listPageUrl = baseUrl + "&start=" + start;
            log.info("목록 페이지 URL: {}", listPageUrl);
            driver.get(listPageUrl);

            try {
                wait.until(ExpectedConditions.presenceOfElementLocated(By.cssSelector("ul.list_news")));
            } catch (TimeoutException e) {
                log.info("뉴스 목록을 찾을 수 없습니다. 크롤링을 종료합니다.");
                break;
            }

            List<WebElement> liElements = driver.findElements(By.cssSelector("ul.list_news li.bx"));
            log.info("페이지 {}에서 뉴스 개수: {}", start, liElements.size());
            if (liElements.isEmpty()) {
                isLastPage = true;
                break;
            }

            List<NewsItem> basicNewsItems = extractBasicNewsItems(liElements);

            for (NewsItem item : basicNewsItems) {
                enrichNewsItem(driver, wait, item);
                log.info("수집 뉴스: 제목: {}, 작성시간: {}", item.getTitle(), item.getPublishedDate());

                if (!isNewsItemValid(item)) {
                    log.info("필수 추가 정보 부족으로 뉴스 건너뜀: {}", item.getUrl());
                    continue;
                }

                if (item.getPublishedDate() != null) {
                    LocalDateTime publishedTime = LocalDateTime.parse(item.getPublishedDate(), DATE_FORMATTER);
                    Instant publishedInstant = publishedTime.atZone(ZoneId.systemDefault()).toInstant();
                    if (publishedInstant.isBefore(thresholdTime)) {
                        log.info("오래된 뉴스 발견 (작성시간: {}) -> 이후 기사들은 수집하지 않습니다.", item.getPublishedDate());
                        stopCrawling = true;
                        break;
                    }
                }
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

        log.info("수집된 뉴스 항목:");
        for (NewsItem news : collectedNews) {
            log.info(news.toString());
        }
    }

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

    private void enrichNewsItem(WebDriver driver, WebDriverWait wait, NewsItem item) {
        try {
            driver.get(item.getUrl());
            wait.until(ExpectedConditions.presenceOfElementLocated(By.tagName("body")));
            String articleHtml = driver.getPageSource();
            org.jsoup.nodes.Document doc = org.jsoup.Jsoup.parse(articleHtml);

            String content = ArticleCleaner.extractMeaningfulContent(articleHtml);
            if (content == null || content.trim().isEmpty()) {
                org.jsoup.nodes.Element articleElem = doc.getElementById("dic_area");
                if (articleElem != null) {
                    content = articleElem.text().trim();
                }
            }
            item.setContent(content);

            org.jsoup.nodes.Element metaDesc = doc.selectFirst("meta[property=og:description]");
            if (metaDesc != null) {
                item.setDescription(metaDesc.attr("content"));
            }

            org.jsoup.nodes.Element metaImage = doc.selectFirst("meta[property=og:image]");
            if (metaImage != null) {
                item.setNewsImage(metaImage.attr("content"));
            }

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

            List<WebElement> timeElems = driver.findElements(By.cssSelector("span.media_end_head_info_datestamp_time._ARTICLE_DATE_TIME"));
            if (!timeElems.isEmpty()) {
                String publishedStr = timeElems.get(0).getAttribute("data-date-time");
                item.setPublishedDate(publishedStr);
            }
        } catch (Exception ex) {
            log.error("기사 페이지 파싱 중 오류 발생 ({}): {}", item.getUrl(), ex.getMessage());
            item.setContent("");
        }
    }

    private boolean isNewsItemValid(NewsItem item) {
        return !((item.getContent() == null || item.getContent().trim().isEmpty()) &&
                (item.getDescription() == null || item.getDescription().trim().isEmpty()) &&
                (item.getNewsImage() == null || item.getNewsImage().trim().isEmpty()) &&
                (item.getPress() == null || item.getPress().trim().isEmpty()) &&
                (item.getPressLogo() == null || item.getPressLogo().trim().isEmpty()) &&
                (item.getPublishedDate() == null || item.getPublishedDate().trim().isEmpty()));
    }
}
