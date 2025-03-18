package newstock.exception;

import lombok.extern.slf4j.Slf4j;
import newstock.common.dto.ExceptionResponse;
import newstock.exception.type.BusinessException;
import newstock.exception.type.DbException;
import newstock.exception.type.ExternalApiException;
import newstock.exception.type.ValidationException;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {

    @ExceptionHandler(ValidationException.class)
    public ResponseEntity<ExceptionResponse> handleValidationException(ValidationException ex) {

        return ResponseEntity
                .status(ex.getExceptionCode().getHttpStatus())
                .body(ExceptionResponse.of(ex.getExceptionCode()));
    }

    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<ExceptionResponse> handleBusinessException(BusinessException ex) {

        return ResponseEntity
                .status(ex.getExceptionCode().getHttpStatus())
                .body(ExceptionResponse.of(ex.getExceptionCode()));
    }

    @ExceptionHandler(DbException.class)
    public ResponseEntity<ExceptionResponse> handleDbException(DbException ex) {

        return ResponseEntity
                .status(ex.getExceptionCode().getHttpStatus())
                .body(ExceptionResponse.of(ex.getExceptionCode()));
    }

    @ExceptionHandler(ExternalApiException.class)
    public ResponseEntity<ExceptionResponse> handleExternalApiException(ExternalApiException ex) {

        return ResponseEntity
                .status(ex.getExceptionCode().getHttpStatus())
                .body(ExceptionResponse.of(ex.getExceptionCode()));
    }
}
