package newstock.domain.notification.repository;

import newstock.domain.notification.entity.UserNotification;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface UserNotificationRepository extends JpaRepository<UserNotification, Integer> {

    @Query("select un from UserNotification un where un.userId = :userId")
    List<UserNotification> findAllByUserId(Integer userId);
}
