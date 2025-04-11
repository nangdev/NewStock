package newstock.common.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Api<T> {

    private static final String SUCCESS_MESSAGE = "정상 처리";
    private static Api<Void> SUCCESS = new Api<>(SUCCESS_MESSAGE);

    private String message;
    private T data;

    public Api(String message) {
        this.message = message;
    }

    public static Api<Void> ok() {
        return SUCCESS;
    }

    public static <T> Api<T> ok(T data) {
        return new Api<>(SUCCESS_MESSAGE, data);
    }

    public static Api<Integer> ERROR(String msg, int code) {
        return new Api<>(msg ,code);
    }

}
