package newstock.domain.news.util;

import java.util.*;

public class CompanyKeywordUtil {

    private static final Map<String, List<String>> COMPANY_KEYWORDS;

    static {
        Map<String, List<String>> map = new HashMap<>();
        map.put("삼성전자", Arrays.asList("삼성전자", "삼전", "s전자"));
        map.put("SK하이닉스", Arrays.asList("SK하이닉스", "하이닉스", "하닉"));
        map.put("LG에너지솔루션", Arrays.asList("LG에너지솔루션", "LG에너지", "에너지솔루션", "엘지에너지솔루션", "엘지엔솔", "LG엔솔"));
        map.put("삼성바이오로직스", Arrays.asList("삼성바이오로직스", "삼바이오", "삼바이로직스", "삼바이로"));
        map.put("현대차", Arrays.asList("현대차", "현대", "현대자동차"));
        map.put("기아", Arrays.asList("기아", "기아차"));
        map.put("셀트리온", Arrays.asList("셀트리온", "셀트"));
        map.put("KB금융", Arrays.asList("KB금융", "KB"));
        map.put("NAVER", Arrays.asList("NAVER", "네이버"));
        map.put("HD현대중공업", Arrays.asList("HD현대중공업", "현중", "현대중공업"));
        map.put("신한지주", Arrays.asList("신한지주", "신한"));
        map.put("현대모비스", Arrays.asList("현대모비스", "모비스"));
        map.put("POSCO홀딩스", Arrays.asList("POSCO홀딩스", "포스코홀딩스", "포스코", "포홀"));
        map.put("삼성물산", Arrays.asList("삼성물산", "삼물"));
        map.put("메리츠금융지주", Arrays.asList("메리츠금융지주", "메리츠"));
        map.put("고려아연", Arrays.asList("고려아연", "아연"));
        map.put("삼성생명", Arrays.asList("삼성생명", "삼생"));
        map.put("LG화학", Arrays.asList("LG화학", "엘지화학", "엘화"));
        map.put("삼성화재", Arrays.asList("삼성화재", "삼화재"));
        map.put("SK이노베이션", Arrays.asList("SK이노베이션", "스크이노", "SK이노"));
        map.put("삼성SDI", Arrays.asList("삼성SDI", "삼스디", "SDI"));
        map.put("카카오", Collections.singletonList("카카오"));
        map.put("한화에어로스페이스", Arrays.asList("한화에어로스페이스", "한화에어"));
        map.put("HD한국조선해양", Arrays.asList("HD한국조선해양", "한국조선해양", "한조해"));
        map.put("하나금융지주", Arrays.asList("하나금융지주", "하나금융"));
        map.put("HMM", Arrays.asList("HMM", "흠슬라", "현대상선"));
        map.put("크래프톤", Arrays.asList("크래프톤", "크래프"));
        map.put("HD현대일렉트릭", Arrays.asList("HD현대일렉트릭", "현대일렉", "현대일렉트릭"));
        map.put("LG전자", Arrays.asList("LG전자", "엘지", "엘전", "엘지전자"));
        map.put("KT&G", Arrays.asList("KT&G", "케이티앤지", "담배인삼공사"));
        map.put("한국전력", Arrays.asList("한국전력", "한전", "전력"));
        map.put("SK텔레콤", Arrays.asList("SK텔레콤", "SK텔", "SKT", "스크트"));
        map.put("한화오션", Arrays.asList("한화오션", "한화오"));
        map.put("두산에너빌리티", Arrays.asList("두산에너빌리티", "두산에너", "구)두산중공업"));
        map.put("기업은행", Arrays.asList("기업은행", "기은", "IBK"));
        map.put("LG", Arrays.asList("LG", "LG지주", "엘지"));
        map.put("우리금융지주", Arrays.asList("우리금융지주", "우리금융"));
        map.put("KT", Arrays.asList("KT", "케이티"));
        map.put("포스코퓨처엠", Arrays.asList("포스코퓨처엠", "포퓨엠", "포스코퓨쳐엠", "포스코"));
        map.put("SK스퀘어", Arrays.asList("SK스퀘어", "스퀘어", "SK", "Sq"));
        COMPANY_KEYWORDS = Collections.unmodifiableMap(map);
    }

    public static boolean isTitleContainsCompanyName(String title, String companyName) {
        if (title == null || companyName == null) return false;
        // 비교를 위해 제목과 키워드를 소문자 및 공백 제거 후 정규화
        String normalizedTitle = title.toLowerCase().replaceAll("\\s+", "");
        List<String> keywords = COMPANY_KEYWORDS.get(companyName);
        if (keywords == null || keywords.isEmpty()) {
            keywords = Collections.singletonList(companyName);
        }
        return keywords.stream()
                .map(kw -> kw.toLowerCase().replaceAll("\\s+", ""))
                .anyMatch(normalizedKeyword -> normalizedTitle.contains(normalizedKeyword));
    }

}
