package newstock.exception;

import lombok.extern.slf4j.Slf4j;
import newstock.common.dto.Api;
import newstock.exception.type.InternalException;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {

    @ExceptionHandler(InternalException.class)
    public ResponseEntity<Api<Integer>> handleInternalException(InternalException ex) {
        log.error("[{}] 예외 발생: {} (코드: {})",
                ex.getClass().getSimpleName(), ex.getMessage(),
                ex.getExceptionCode().getCode());

        ExceptionCode ec = ex.getExceptionCode();

        return ResponseEntity.status(ec.getStatus()).body(Api.ERROR(ec.getMessage(), ec.getCode()));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Api<Integer>> handleGlobalException(Exception ex) {
        log.error("예상치 못한 오류 발생:", ex);

        ExceptionCode ec = ExceptionCode.EXTERNAL_API_ERROR;

        return ResponseEntity.status(ec.getStatus()).body(Api.ERROR(ec.getMessage(), ec.getCode()));
    }
}