package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class UserResponse {

    private Long userId;
    private String email;
    private String username;
    private String nickname;

}
