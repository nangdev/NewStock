package newstock.controller.request;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class AddUserStockRequest {

    private Integer stockId;

}
