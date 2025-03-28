package newstock.exception.type;

import lombok.Getter;
import newstock.exception.ExceptionCode;

@Getter
public class ExternalApiException extends InternalException {

    public ExternalApiException(ExceptionCode exceptionCode) {
        super(exceptionCode);
    }

    public ExternalApiException(ExceptionCode exceptionCode, String message) {
        super(exceptionCode, message);
    }
}
