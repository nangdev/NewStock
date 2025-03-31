package newstock.domain.notification.repository;

import newstock.domain.notification.entity.UserNotification;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

@Repository
public interface UserNotificationRepository extends JpaRepository<UserNotification, Integer>, UserNotificationRepositoryCustom {

    @Query("select un from UserNotification un where un.userId = :userId and un.notificationId = :notificationId")
    UserNotification findByUserIdAndNotificationId (int userId, int notificationId);

    @Modifying
    @Query("delete from UserNotification un where un.userId = :userId and un.notificationId = :notificationId")
    void deleteByUserIdAndNotificationId(int userId, int notificationId);

}
