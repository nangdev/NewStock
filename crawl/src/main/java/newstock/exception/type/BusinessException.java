package newstock.exception.type;

import lombok.Getter;
import newstock.exception.ExceptionCode;

@Getter
public class BusinessException extends RuntimeException {

    private final ExceptionCode exceptionCode;

    public BusinessException(ExceptionCode exceptionCode) {
        super(exceptionCode.getMessage());
        this.exceptionCode = exceptionCode;
    }

    public BusinessException(ExceptionCode exceptionCode, String message) {
        super(message);
        this.exceptionCode = exceptionCode;
    }
}
