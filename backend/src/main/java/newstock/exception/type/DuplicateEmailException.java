package newstock.exception.type;

import newstock.exception.ExceptionCode;

public class DuplicateEmailException extends ValidationException {
    public DuplicateEmailException() {
        super(ExceptionCode.DUPLICATE_EMAIL);
    }
}
