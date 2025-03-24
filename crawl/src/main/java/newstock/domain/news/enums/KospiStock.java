package newstock.domain.news.enums;

public enum KospiStock {
    SAMSUNG_ELECTRONICS("삼성전자", "005930"),
    SK_HYUNIX("SK하이닉스", "000660"),
    LG_ENERGY_SOLUTION("LG에너지솔루션", "373220"),
    SAMSUNG_BIOLOGICS("삼성바이오로직스", "207940"),
    HYUNDAI_MOTOR("현대차", "005380"),
    KIA("기아", "000270"),
    CELLTRION("셀트리온", "068270"),
    KB_FINANCIAL("KB금융", "105560"),
    NAVER("NAVER", "035420"),
    HD_MODERN_HEAVY_INDUSTRIES("HD현대중공업", "329180"),
    SHINHAN_FINANCIAL("신한지주", "055550"),
    HYUNDAI_MOBIS("현대모비스", "012330"),
    POSCO_HOLDINGS("POSCO홀딩스", "005490"),
    SAMSUNG_C_AND_T("삼성물산", "028260"),
    MERITZ_FINANCIAL("메리츠금융지주", "138040"),
    KOREA_ZINC("고려아연", "010130"),
    SAMSUNG_LIFE("삼성생명", "032830"),
    LG_CHEM("LG화학", "051910"),
    SAMSUNG_FIRE("삼성화재", "000810"),
    SK_INNOVATION("SK이노베이션", "096770"),
    SAMSUNG_SDI("삼성SDI", "006400"),
    KAKAO("카카오", "035720"),
    HANWHA_AERO("한화에어로스페이스", "012450"),
    HD_KOREA_SHIPBUILDING("HD한국조선해양", "009540"),
    HANA_FINANCIAL("하나금융지주", "086790"),
    HMM("HMM", "011200"),
    KRAFTON("크래프톤", "259960"),
    HD_HYUNDAI_ELECTRIC("HD현대일렉트릭", "267260"),
    LG_ELECTRONICS("LG전자", "066570"),
    KT_G("KT&G", "033780"),
    KOREA_ELECTRIC_POWER("한국전력", "015760"),
    SK_TELECOM("SK텔레콤", "017670"),
    HANWHA_OCEAN("한화오션", "042660"),
    DOOSAN_ENERBILITY("두산에너빌리티", "034020"),
    IBK("기업은행", "024110"),
    LG("LG", "003550"),
    WOORI_FINANCIAL("우리금융지주", "316140"),
    KT("KT", "030200"),
    POSCO_FUTURE_M("포스코퓨처엠", "003670"),
    SK_SQUARE("SK스퀘어", "402340"),
    SAMSUNG_HEAVY_INDUSTRIES("삼성중공업", "010140"),
    KAKAO_BANK("카카오뱅크", "323410"),
    HYUNDAI_GLOVIS("현대글로비스", "086280"),
    SK("SK", "034730"),
    SAMSUNG_SDS("삼성에스디에스", "018260"),
    YUHAN("유한양행", "000100"),
    SAMSUNG_ELECTRO_MECHANICS("삼성전기", "009150"),
    KOREAN_AIR("대한항공", "003490"),
    SK_BIOPHARM("SK바이오팜", "326030"),
    HANMI_SEMICON("한미반도체", "042700");

    private final String name;
    private final String code;

    KospiStock(String name, String code) {
        this.name = name;
        this.code = code;
    }

    public String getName() {
        return name;
    }

    public String getCode() {
        return code;
    }

}

