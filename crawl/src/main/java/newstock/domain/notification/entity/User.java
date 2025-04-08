package newstock.domain.notification.entity;


import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;

import java.time.LocalDateTime;

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


}

