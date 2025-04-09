package newstock.domain.news.util;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;

public class ArticleCleaner {

    /**
     * 기사 HTML 전체에서 id가 "dic_area"인 article 태그 내부의 이미지 이후부터의 내용만 추출합니다.
     * (불필요한 태그(table, script, style, em.img_desc 등)는 제거합니다.)
     * 그리고 각 <br> 태그마다 개행 문자를 두 개("\n\n")만 남도록 처리합니다.
     *
     * @param html 기사 페이지의 HTML 전체
     * @return 이미지 이후부터 추출한 기사 본문 텍스트
     */
    public static String extractMeaningfulContent(String html) {
        Document doc = Jsoup.parse(html);
        Element article = doc.getElementById("dic_area");
        if (article == null) {
            return "";
        }

        article.select("table, script, style, em.img_desc, div[style*='border:1px solid #e6e6e6']").remove();

        // article 내에서 첫 번째 img 요소를 찾습니다.
        Element firstImg = article.select("img").first();
        if (firstImg == null) {
            // 이미지가 없으면 전체 텍스트를 반환
            return article.text().trim();
        }

        // 이미지가 포함된 최상위 부모 요소를 찾습니다.
        Element imgContainer = firstImg.parent();
        while (imgContainer != null && !imgContainer.parent().equals(article)) {
            imgContainer = imgContainer.parent();
        }
        if (imgContainer == null) {
            imgContainer = firstImg;
        }

        // 이미지 이후의 텍스트를 순회하면서 추출 (개행 처리는 별도 메서드로 수행)
        StringBuilder sb = new StringBuilder();
        boolean startCollecting = false;
        for (Node node : article.childNodes()) {
            if (!startCollecting) {
                // imgContainer와 일치하면 그 이후부터 수집 시작
                if (node.equals(imgContainer)) {
                    startCollecting = true;
                    continue; // 이미지 컨테이너 자체는 제외 (필요에 따라 포함 가능)
                }
            } else {
                appendTextWithLineBreaks(node, sb);
                sb.append(" "); // 각 노드별로 약간의 공백 추가 (옵션)
            }
        }
        String result = sb.toString().trim();
        // 3개 이상의 연속된 개행 문자가 있으면 두 개의 개행("\n\n")으로 치환
        result = result.replaceAll("(\\n){3,}", "\n\n");
        return result;
    }

    private static void appendTextWithLineBreaks(Node node, StringBuilder sb) {
        if (node.nodeName().equalsIgnoreCase("br")) {
            // <br> 태그를 만나면 바로 개행 문자를 추가합니다.
            sb.append("\n");
        } else if (node instanceof TextNode) {
            sb.append(((TextNode) node).text());
        } else if (node instanceof Element) {
            for (Node child : ((Element) node).childNodes()) {
                appendTextWithLineBreaks(child, sb);
            }
        }
    }
}
