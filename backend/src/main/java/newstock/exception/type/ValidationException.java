package newstock.exception.type;

import lombok.Getter;
import newstock.exception.ExceptionCode;

@Getter
public class ValidationException extends InternalException {
    public ValidationException(ExceptionCode exceptionCode) {
        super(exceptionCode);
    }

    public ValidationException(ExceptionCode exceptionCode, String message) {
        super(exceptionCode, message);
    }
}
