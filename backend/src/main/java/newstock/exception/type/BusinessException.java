package newstock.exception.type;

import lombok.Getter;
import newstock.exception.ExceptionCode;

@Getter
public class BusinessException extends InternalException {

    public BusinessException(ExceptionCode exceptionCode) {
        super(exceptionCode);
    }

    public BusinessException(ExceptionCode exceptionCode, String message) {
        super(exceptionCode, message);
    }
}
