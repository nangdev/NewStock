package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.request.UpdateUserStockListRequest;
import newstock.controller.response.StockListResponse;
import newstock.controller.response.UserStockListResponse;
import newstock.domain.stock.dto.StockInfoDto;
import newstock.domain.stock.service.StockService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/stock")
public class StockController {

    private final StockService stockService;

    @GetMapping
    public ResponseEntity<Api<StockListResponse>> getStockList() {
        return ResponseEntity.ok(Api.ok(StockListResponse.of(stockService.getStockList())));
    }

    @GetMapping("/interest")
    public ResponseEntity<Api<UserStockListResponse>> getUserStockList(@AuthenticationPrincipal Integer userId) {
        return ResponseEntity.ok(Api.ok(UserStockListResponse.of(stockService.getUserStockList(userId))));
    }

    @PutMapping("/interest")
    public ResponseEntity<Api<Void>> updateUserStockList(@AuthenticationPrincipal Integer userId, @RequestBody UpdateUserStockListRequest req){
        stockService.updateUserStockList(userId, req.getStockIdList());
        return ResponseEntity.ok(Api.ok());
    }

    @GetMapping("/info/{stockId}")
    public ResponseEntity<Api<StockInfoDto>> getStockInfo(@PathVariable Integer stockId){
        return ResponseEntity.ok(Api.ok(stockService.getStockInfo(stockId)));
    }

}
