package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class UserResponse {

    private String name;
    private String nickName;
    private String email;

    public static UserResponse of(String name, String nickName, String email) {

        return UserResponse.builder()
                .name(name)
                .nickName(nickName)
                .email(email)
                .build();
    }
}
