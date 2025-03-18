package newstock.common.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import newstock.exception.ExceptionCode;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExceptionResponse {

    private int code;
    private String message;

    public static ExceptionResponse of(ExceptionCode exceptionCode) {
        return ExceptionResponse.builder()
                .code(exceptionCode.getCode())
                .message(exceptionCode.getMessage())
                .build();
    }
}
