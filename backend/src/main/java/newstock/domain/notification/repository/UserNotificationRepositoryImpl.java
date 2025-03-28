package newstock.domain.notification.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.news.entity.QNews;
import newstock.domain.notification.dto.NewsNotificationDto;
import newstock.domain.notification.dto.StockNotificationDto;
import newstock.domain.notification.dto.UserNotificationDto;
import newstock.domain.notification.entity.QNotification;
import newstock.domain.notification.entity.QUserNotification;
import newstock.domain.stock.entity.QStock;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
@RequiredArgsConstructor
public class UserNotificationRepositoryImpl implements UserNotificationRepositoryCustom{

    private final JPAQueryFactory queryFactory;

    @Override
    public List<UserNotificationDto> findUserNotificationsWithDetails(Integer userId) {
        QUserNotification un = QUserNotification.userNotification;
        QNotification notif = QNotification.notification;
        QNews news = QNews.news;
        QStock stock = QStock.stock;


        return queryFactory
                .select(Projections.constructor(UserNotificationDto.class,
                        un.unId,
                        Projections.constructor(NewsNotificationDto.class,
                                news.newsId,
                                news.title,
                                news.publishedDate
                        ).as("newsInfo"),
                        Projections.constructor(StockNotificationDto.class,
                                stock.stockId,
                                stock.stockCode,
                                stock.stockName
                        ).as("stockInfo"),
                        un.isRead
                ))
                .from(un)
                .leftJoin(notif).on(un.notificationId.eq(notif.notificationId))
                .leftJoin(news).on(notif.newsId.eq(news.newsId))
                .leftJoin(stock).on(notif.stockId.eq(stock.stockId))
                .where(un.userId.eq(userId))
                .fetch();
    }

}
