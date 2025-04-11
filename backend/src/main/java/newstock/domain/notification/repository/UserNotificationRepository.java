package newstock.domain.notification.repository;

import newstock.domain.notification.entity.UserNotification;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserNotificationRepository extends JpaRepository<UserNotification, Integer>, UserNotificationRepositoryCustom {

    List<UserNotification> findAllByUserId(Integer userId);
}
