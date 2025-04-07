package newstock.domain.user.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import newstock.controller.request.UserRequest;
import org.hibernate.annotations.SQLDelete;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;
import java.time.LocalDateTime;

@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@EntityListeners(AuditingEntityListener.class)
@SQLDelete(sql = "UPDATE users SET is_activated = false WHERE user_id = ?")
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer userId;

    @Column(unique = true)
    private String email;

    @Column
    private String password;

    @Column(nullable = false)
    private String nickname;

    @Column
    private String refreshToken;

    @Column(unique = true)
    private Long kakaoId;

    @Column
    private String socialProvider; // "kakao", null (일반가입)

    @Column
    private String fcmToken;

    @Column(nullable = false)
    private Byte role; // 유저 권한 0이면 NEW(신규 회원), 1이면 USER(기존 유저)

    @Column(nullable = false)
    private boolean activated;

    @CreatedDate
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @LastModifiedDate
    private LocalDateTime updatedAt;

    public static User of(UserRequest userRequest, String encodedPassword) {
        return User.builder()
                .email(userRequest.getEmail())
                .password(encodedPassword)
                .nickname(userRequest.getNickname())
                .role((byte) 0)
                .activated(true)
                .build();
    }

    public static User ofKakao(Long kakaoId, String email, String nickname) {
        return User.builder()
                .kakaoId(kakaoId)
                .email(email)
                .nickname(nickname)
                .socialProvider("kakao")
                .role((byte) 0)
                .activated(true)
                .build();
    }

    public void reactivate(UserRequest request, String encodedPassword) {
        this.password = encodedPassword;
        this.nickname = request.getNickname();
        this.activated = true;
    }
}