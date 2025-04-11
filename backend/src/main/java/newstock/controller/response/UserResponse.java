package newstock.controller.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import newstock.domain.user.entity.User;

@Getter
@Builder
@AllArgsConstructor
public class UserResponse {

    private Integer userId;
    private String email;
    private String nickname;
    private Byte role;

    public static UserResponse of(User user) {
        return UserResponse.builder()
                .userId(user.getUserId())
                .email(user.getEmail())
                .nickname(user.getNickname())
                .role(user.getRole())
                .build();
    }
}
