package newstock.domain.user.entity;

import jakarta.persistence.*;
import lombok.*;
import newstock.controller.request.UserRequest;

@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Table(name="users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer userId; // 유저 고유 아이디

    @Column(unique = true)
    private String email; // 유저 로그인 아이디(이메일)

    @Column
    private String password; // 유저 비밀번호

    @Column(nullable = false)
    private String userName; // 유저 실명

    @Column(nullable = false)
    private String nickname; // 유저 닉네임

    @Column
    private String accessToken; // 어세스토큰

    @Column
    private String refreshToken; // 리프레시 토큰

    @Column
    private String refreshTokenExpires; // 리프레시 토큰 만료 기간

    @Column
    private String socialProvider; // 소셜 로그인 여부

    @Column
    private String fcmToken; // 알림 토큰

    @Column(nullable = false)
    private Byte role; // 유저 권한 0이면 NEW(신규 회원), 1이면 USER(기존 유저)

    public static User of(UserRequest userRequest, String encodedPassword) {
        return User.builder()
                .email(userRequest.getEmail())
                .password(encodedPassword)
                .userName(userRequest.getUserName() )
                .nickname(userRequest.getNickname())
                .role((byte) 0)
                .build();
    }
}




