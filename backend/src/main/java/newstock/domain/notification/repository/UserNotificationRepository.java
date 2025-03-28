package newstock.domain.notification.repository;

import newstock.domain.notification.entity.UserNotification;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserNotificationRepository extends JpaRepository<UserNotification, Integer>, UserNotificationRepositoryCustom {

}
