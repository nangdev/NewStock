package newstock.domain.stock.repository;

import newstock.domain.notification.dto.NotificationResultDto;


public interface UserStockRepositoryCustom {
    NotificationResultDto findUsersAndNewsById(Integer stockId, Integer newsId);
}
