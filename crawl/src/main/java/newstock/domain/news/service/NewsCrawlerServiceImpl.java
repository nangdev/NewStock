package newstock.domain.news.service;

import com.microsoft.playwright.*;
import com.microsoft.playwright.options.LoadState;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.news.dto.NewsItem;
import newstock.kafka.request.NewsCrawlerRequest;
import newstock.domain.news.util.ArticleCleaner;
import newstock.domain.news.util.CompanyKeywordUtil;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Service
@Slf4j
@RequiredArgsConstructor
public class NewsCrawlerServiceImpl implements NewsCrawlerService {

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    // 스케줄러 기준 시간 대비 3분 전보다 오래된 뉴스는 수집하지 않음 (탐색 종료)
    private static final Duration LOWER_BOUND_DURATION = Duration.ofMinutes(3);
    // 스크롤 후 대기 시간 (밀리초)
    private static final int SCROLL_TIMEOUT_MS = 2000;
    // 최대 스크롤 횟수 (무한 스크롤이 계속되는 경우를 방지)
    private static final int MAX_SCROLL_COUNT = 5;

    /**
     * 주어진 종목명에 대해 Playwright를 사용해 뉴스 목록을 무한 스크롤 방식으로 탐색하고,
     * 스케줄러 기준 시간으로부터 3분 전 이후(3분 전 포함)의 뉴스만 수집하여 리스트로 반환합니다.
     * 위에서부터 탐색하다가 최초로 스케줄러 기준 시간 3분 전보다 오래된 뉴스가 나오면 탐색을 중단합니다.
     * 단, 뉴스 제목에 회사명(전체 또는 줄임말)이 포함되어 있지 않으면 수집하지 않습니다.
     */
    @Override
    public List<NewsItem> fetchNews(NewsCrawlerRequest newsCrawlerRequest) {
        List<NewsItem> collectedNews = new ArrayList<>();
        // 중복 처리를 위해 URL 저장용 셋
        Set<String> processedUrls = new HashSet<>();

        // 기본 URL 구성 (검색어, 정렬 조건 등)
        String baseUrl = "https://search.naver.com/search.naver?where=news&query="
                + newsCrawlerRequest.getStockName()
                + "&sm=tab_opt&sort=1&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=0"
                + "&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall"
                + "&is_sug_officeid=0&office_category=0&service_area=0";

        // 스케줄러 기준 시간과 수집 범위 계산 (cutoff: 스케줄러 기준 3분 전)
        Instant schedulerTime = Instant.parse(newsCrawlerRequest.getSchedulerTime());
        Instant cutoff = schedulerTime.minus(LOWER_BOUND_DURATION);

        try (Playwright playwright = Playwright.create()) {
            Browser browser = playwright.chromium().launch(
                    new BrowserType.LaunchOptions().setHeadless(true)
            );
            BrowserContext context = browser.newContext(
                    new Browser.NewContextOptions()
                            .setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
                                    "(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36")
                            .setLocale("ko-KR")
                            .setViewportSize(1280, 720)
            );
            Page page = context.newPage();

            // 뉴스 목록 페이지로 이동
            page.navigate(baseUrl);
            page.waitForSelector("ul.list_news", new Page.WaitForSelectorOptions().setTimeout(5000));

            boolean stopCrawling = false;
            int scrollCount = 0;
            int previousNewsCount = 0;

            while (!stopCrawling && scrollCount < MAX_SCROLL_COUNT) {
                String htmlContent = page.content();
                Document listDoc = Jsoup.parse(htmlContent);
                Elements liElements = listDoc.select("ul.list_news li.bx");

                if (liElements.isEmpty()) {
                    log.info("뉴스 목록을 찾을 수 없습니다. (종목: {})", newsCrawlerRequest.getStockName());
                    break;
                }

                // 뉴스 아이템 추출
                List<NewsItem> basicNewsItems = extractBasicNewsItems(liElements);
                for (NewsItem item : basicNewsItems) {
                    // 제목에 회사명(전체 또는 줄임말)이 포함되어 있지 않으면 건너뜁니다.
                    if (!CompanyKeywordUtil.isTitleContainsCompanyName(item.getTitle(), newsCrawlerRequest.getStockName())) {
                        continue;
                    }
                    // 중복 처리
                    if (processedUrls.contains(item.getUrl())) {
                        continue;
                    }
                    processedUrls.add(item.getUrl());

                    // 실제 기사 페이지에서 추가 정보 수집
                    enrichNewsItemWithPlaywright(page, item);
                    if (!isNewsItemValid(item)) {
                        continue;
                    }

                    // 작성 시간 확인: 스케줄러 기준 3분 전보다 오래된 뉴스가 나오면 탐색 종료
                    if (item.getPublishedDate() != null) {
                        try {
                            LocalDateTime publishedTime = LocalDateTime.parse(item.getPublishedDate(), DATE_FORMATTER);
                            Instant publishedInstant = publishedTime.atZone(ZoneId.of("Asia/Seoul")).toInstant();
                            if (publishedInstant.isBefore(cutoff)) {
                                log.info("임계시간(3분 전)보다 오래된 뉴스 발견. 이후 뉴스는 수집하지 않습니다. (종목: {})", newsCrawlerRequest.getStockName());
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

                // 만약 새로 추가된 뉴스가 없다면 스크롤 종료
                if (processedUrls.size() == previousNewsCount) {
                    log.info("새로운 뉴스 아이템이 더 이상 로드되지 않습니다. (종목: {})", newsCrawlerRequest.getStockName());
                    break;
                }
                previousNewsCount = processedUrls.size();

                if (stopCrawling) {
                    break;
                }
                // 스크롤 내리기
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)");
                page.waitForTimeout(SCROLL_TIMEOUT_MS);
                scrollCount++;
            }

            context.close();
            browser.close();
        } catch (Exception e) {
            log.error("뉴스 크롤링 중 오류 발생: {}", e.getMessage(), e);
        }
        return collectedNews;
    }

    /**
     * 뉴스 목록 페이지에서 기본 뉴스 아이템(제목, URL 등)을 추출합니다.
     */
    private List<NewsItem> extractBasicNewsItems(Elements liElements) {
        List<NewsItem> basicNewsItems = new ArrayList<>();
        for (Element li : liElements) {
            try {
                Elements infoLinkElems = li.select("div.info_group a:contains(네이버뉴스)");
                if (infoLinkElems.isEmpty()) {
                    continue;
                }
                String newsLink = infoLinkElems.first().attr("href");
                if (newsLink == null || !newsLink.contains("n.news.naver.com")) {
                    continue;
                }
                Element titleElem = li.selectFirst("a.news_tit");
                String title = (titleElem != null) ? titleElem.attr("title") : "";
                if ((title == null || title.trim().isEmpty()) && titleElem != null) {
                    title = titleElem.text();
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
     * Playwright를 사용해 실제 기사 페이지에서 추가 정보를 수집합니다.
     */
    private void enrichNewsItemWithPlaywright(Page page, NewsItem item) {
        try {
            page.navigate(item.getUrl());
            page.waitForLoadState(LoadState.NETWORKIDLE, new Page.WaitForLoadStateOptions().setTimeout(5000));
            String articleHtml = page.content();
            Document doc = Jsoup.parse(articleHtml);

            // 본문 추출: ArticleCleaner 활용, 없으면 id="dic_area"에서 추출
            String content = ArticleCleaner.extractMeaningfulContent(articleHtml);
            if (content == null || content.trim().isEmpty()) {
                Element articleElem = doc.getElementById("dic_area");
                if (articleElem != null) {
                    content = articleElem.text().trim();
                }
            }
            item.setContent(content);

            // 메타 태그에서 설명과 대표 이미지 추출
            Element metaDesc = doc.selectFirst("meta[property=og:description]");
            if (metaDesc != null) {
                item.setDescription(metaDesc.attr("content"));
            }
            Element metaImage = doc.selectFirst("meta[property=og:image]");
            if (metaImage != null) {
                item.setNewsImage(metaImage.attr("content"));
            }

            // 언론사 로고 및 이름 추출
            Element logoElem = doc.selectFirst("img.media_end_head_top_logo_img.light_type");
            if (logoElem != null) {
                String pressLogo = logoElem.hasAttr("data-src") ? logoElem.attr("data-src") : logoElem.attr("src");
                item.setPressLogo(pressLogo);
                String pressName = logoElem.hasAttr("alt") ? logoElem.attr("alt") : logoElem.attr("title");
                item.setPress(pressName);
            }

            // 작성 시간 추출
            Element timeElem = doc.selectFirst("span.media_end_head_info_datestamp_time._ARTICLE_DATE_TIME");
            if (timeElem != null) {
                String publishedStr = timeElem.attr("data-date-time");
                item.setPublishedDate(publishedStr);
            }
        } catch (Exception ex) {
            log.error("기사 페이지 파싱 중 오류 발생 ({}): {}", item.getUrl(), ex.getMessage(), ex);
            item.setContent("");
        }
    }

    /**
     * 필수 추가 정보(본문, 설명, 대표 이미지, 언론사, 로고, 작성시간)가 모두 존재하는지 확인합니다.
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
