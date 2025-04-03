package newstock.domain.notification.entity;


import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Table(name="users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer userId;

    @Column(unique = true)
    private String email;

    @Column
    private String password;

    @Column(nullable = false)
    private String userName;

    @Column(nullable = false)
    private String nickname;

    @Column
    private String accessToken;

    @Column
    private String refreshToken;

    @Column
    private String refreshTokenExpires;

    @Column
    private String socialProvider;

    @Column
    private String fcmToken;

    @Column(nullable = false)
    private Byte role; // 유저 권한 0이면 NEW(신규 회원), 1이면 USER(기존 유저)

}

