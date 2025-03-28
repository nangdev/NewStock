package newstock.exception.type;

import lombok.Getter;
import newstock.exception.ExceptionCode;

@Getter
public class DbException extends InternalException {

    public DbException(ExceptionCode exceptionCode) {
        super(exceptionCode);
    }

    public DbException(ExceptionCode exceptionCode, String message) {
        super(exceptionCode, message);
    }
}
