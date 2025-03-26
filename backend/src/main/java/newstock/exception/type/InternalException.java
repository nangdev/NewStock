package newstock.exception.type;

import newstock.exception.ExceptionCode;

public class InternalException extends RuntimeException {

    private final ExceptionCode exceptionCode;

    public InternalException(ExceptionCode exceptionCode) {
      super(exceptionCode.getMessage());
      this.exceptionCode = exceptionCode;
    }

    public InternalException(ExceptionCode exceptionCode, String message) {
      super(message);
      this.exceptionCode = exceptionCode;
    }

    public ExceptionCode getExceptionCode() {
        return exceptionCode;
    }

}
