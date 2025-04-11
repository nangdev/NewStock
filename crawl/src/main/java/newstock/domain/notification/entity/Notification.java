package newstock.domain.notification.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.Data;
import lombok.NoArgsConstructor;


@Entity
@Data
@NoArgsConstructor
public class Notification {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY )
    private Integer notificationId;

    private Integer stockId;

    private Integer newsId;

    public Notification(Integer stockId, Integer newsId) {
        this.stockId = stockId;
        this.newsId = newsId;
    }

}
