package newstock.domain.news.util;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;

public class ArticleCleaner {

    /**
     * 기사 HTML 전체에서 id가 "dic_area"인 article 태그 내부의 모든 텍스트를 추출합니다.
     * (불필요한 태그(table, script, style, em, div[style*='border:1px solid #e6e6e6'],
     * strong.media_end_summary 등)는 제거합니다.)
     * 그리고 각 <br> 태그마다 개행 문자를 두 개("\n\n")만 남도록 처리합니다.
     *
     * @param html 기사 페이지의 HTML 전체
     * @return 추출한 기사 본문 텍스트
     */
    public static String extractMeaningfulContent(String html) {
        Document doc = Jsoup.parse(html);
        Element article = doc.getElementById("dic_area");
        if (article == null) {
            return "";
        }

        // 불필요한 태그 제거 - 모든 em 태그를 포함하도록 수정
        article.select("table, script, style, em, div[style*='border:1px solid #e6e6e6'], strong.media_end_summary").remove();

        // article 내부의 모든 텍스트를 순회하며 추출 (개행 처리는 별도 메서드 사용)
        StringBuilder sb = new StringBuilder();
        for (Node node : article.childNodes()) {
            appendTextWithLineBreaks(node, sb);
            sb.append(" "); // 각 노드별로 공백 추가 (옵션)
        }
        String result = sb.toString().trim();
        // 3개 이상의 연속된 개행 문자가 있으면 두 개의 개행("\n\n")으로 치환
        result = result.replaceAll("(\\n){3,}", "\n\n");
        return result;
    }

    private static void appendTextWithLineBreaks(Node node, StringBuilder sb) {
        if (node.nodeName().equalsIgnoreCase("br")) {
            // <br> 태그를 만나면 개행 문자를 추가
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
