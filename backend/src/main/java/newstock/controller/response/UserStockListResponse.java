package newstock.controller.response;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.stock.dto.UserStockDto;

import java.util.List;

@Builder
@Getter
public class UserStockListResponse {

    private List<UserStockDto> stockList;

    public static UserStockListResponse of(List<UserStockDto> stockList) {
        return UserStockListResponse.builder()
                .stockList(stockList)
                .build();
    }

}
