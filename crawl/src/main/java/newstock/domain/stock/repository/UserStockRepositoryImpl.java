package newstock.domain.stock.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import newstock.domain.news.dto.NotificationNewsDto;
import newstock.domain.notification.dto.NotificationResultDto;
import newstock.domain.notification.dto.UserDto;
import org.springframework.stereotype.Repository;

import static newstock.domain.stock.entity.QUserStock.userStock;
import static newstock.domain.notification.entity.QUser.user;
import static newstock.domain.news.entity.QNews.news;

import java.util.List;

@Repository
@RequiredArgsConstructor
public class UserStockRepositoryImpl implements UserStockRepositoryCustom{

    private final JPAQueryFactory queryFactory;

    @Override
    public NotificationResultDto findUsersAndNewsById(Integer stockId, Integer newsId) {
        List<UserDto> userDtos = queryFactory
                .select(Projections.constructor(UserDto.class,
                        userStock.userId,
                        user.fcmToken))
                .from(user, userStock)
                .where(userStock.stockId.eq(stockId),
                        user.userId.eq(userStock.userId))
                .fetch();

        NotificationNewsDto newsDto = queryFactory
                .select(Projections.constructor(NotificationNewsDto.class,
                        news.newsId,
                        news.title,
                        news.description))
                .from(news)
                .where(news.newsId.eq(newsId))
                .fetchOne();


        return new NotificationResultDto(userDtos, newsDto);
    }

}
