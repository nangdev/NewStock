package newstock.domain.notification.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserNotification {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer unId;

    private Integer notificationId;

    private Integer userId;

    private Byte isRead;

    public static UserNotification of(Integer notificationId, Integer userId) {
        return UserNotification.builder()
                .unId(null)
                .notificationId(notificationId)
                .userId(userId)
                .isRead((byte) 0)
                .build();
    }
}

