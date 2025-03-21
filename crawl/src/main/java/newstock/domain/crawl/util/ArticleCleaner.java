package newstock.domain.crawl.util;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class ArticleCleaner {

    /**
     * 기사 HTML 전체에서 id가 "dic_area"인 article 태그 내부의 의미있는 기사 내용만 추출합니다.
     * 불필요한 태그(table, script, style 등)는 제거하고, <br> 태그는 개행 문자로 변환합니다.
     *
     * @param html 기사 페이지의 HTML 전체
     * @return 정리된 기사 본문 텍스트
     */
    public static String extractMeaningfulContent(String html) {
        Document doc = Jsoup.parse(html);
        Element article = doc.getElementById("dic_area");
        if (article == null) {
            return "";
        }

        // <br> 태그를 개행 문자("\n")로 변환
        for (Element br : article.select("br")) {
            br.after("\n");
        }

        // 필요없는 태그 제거 (table, script, style 등)
        Elements unwanted = article.select("table, script, style");
        unwanted.remove();

        return article.text().trim();
    }
}
