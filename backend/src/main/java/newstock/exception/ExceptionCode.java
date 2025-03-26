package newstock.exception;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum ExceptionCode {

    // Controller에서 검증시 발생할 수 있는 예외 작성
    VALIDATION_ERROR("사용자 입력 값이 검증에 실패했습니다.", 1001),
    DUPLICATE_EMAIL("이미 사용 중인 이메일입니다.", 1002),

    // Service에서 비즈니스 로직 처리시 발생할 수 있는 예외 작성
    BUSINESS_ERROR("비즈니스 로직에서 예외가 발생했습니다.", 2001),

    // Repository에서 데이터베이스 조작시 발생할 수 있는 예외 작성
    DATABASE_ERROR("데이터베이스 조작 과정에서 예외가 발생했습니다.", 3001),
    USER_NOT_FOUND("해당하는 사용자가 존재하지 않습니다.", 3002),
    STOCK_NOT_FOUND("해당하는 주식이 존재하지 않습니다.", 3003),
    NEWS_NOT_FOUND("해당하는 뉴스가 존재하지 않습니다.", 3004),
    NEWS_LETTER_NOT_FOUND("해당하는 뉴스레터가 존재하지 않습니다.", 3005),
    NEWS_SCRAP_NOT_FOUND("해당하는 뉴스 스크랩이 존재하지 않습니다.", 3006),
    NOTIFICATION_NOT_FOUND("해당하는 알림이 존재하지 않습니다.", 3007),
    USER_STOCK_NOT_FOUND("해당하는 유저 주식이 존재하지 않습니다.", 3008),
    USER_NOTIFICATION_NOT_FOUND("해당하는 유저 알림이 존재하지 않습니다.", 3009),
    STOCK_ALREADY_EXISTS("이미 존재하는 주식입니다.", 3010),
    USER_STOCK_ALREADY_EXISTS("이미 존재하는 유저 주식입니다.", 3011),

    // 외부 API 사용시 발생할 수 있는 예외 작성
    EXTERNAL_API_ERROR("외부 API를 호출하는 과정에서 예외가 발생했습니다.", 4001),

    // 인증 및 보안 관련 예외 작성
    TOKEN_INVALID("유효하지 않은 토큰입니다.", 5001),

    // 원인 미상 에러
    INTERNAL_SERVER_ERROR("비상 비상 !! 개발자에게 문의하세요 !!!", 44444444);

    private final String message;
    private final int code;

}
