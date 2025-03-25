package newstock.common.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
public class Api<T> {

    private static final String SUCCESS_MESSAGE = "정상 처리";
    private static Api<Void> SUCCESS = new Api<>(true, SUCCESS_MESSAGE);

    private boolean success;
    private String message;
    private T data;

    public Api(boolean success, String message) {
        this.success = success;
        this.message = message;
    }

    public static Api<Void> ok() {
        return SUCCESS;
    }

    public static <T> Api<T> ok(T data) {
        return new Api<>(true, SUCCESS_MESSAGE, data);
    }

    public static Api<Integer> ERROR(String msg, int code) {
        return new Api<>(false, msg ,code);
    }

}
