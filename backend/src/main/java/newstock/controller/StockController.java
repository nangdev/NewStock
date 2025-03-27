package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.request.UpdateUserStockListRequest;
import newstock.controller.response.StockListResponse;
import newstock.controller.response.UserStockListResponse;
import newstock.domain.stock.dto.StockInfoDto;
import newstock.domain.stock.service.StockService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/stock")
public class StockController {

    private final StockService stockService;

    @GetMapping
    public Api<StockListResponse> getStockList() {
        return Api.ok(StockListResponse.of(stockService.getStockList()));
    }

    @GetMapping("/interest")
    public Api<UserStockListResponse> getUserStockList(@AuthenticationPrincipal Integer userId) {
        return Api.ok(UserStockListResponse.of(stockService.getUserStockList(userId)));
    }

    @PutMapping("/interest")
    public Api<Void> updateUserStockList(@AuthenticationPrincipal Integer userId, @RequestBody UpdateUserStockListRequest req){
        stockService.updateUserStockList(userId, req.getStockCodeList());
        return Api.ok();
    }

    @GetMapping("/info/{stockCode}")
    public Api<StockInfoDto> getStockInfo(@PathVariable int stockCode){
        return Api.ok(stockService.getStockInfo(stockCode));
    }

}
