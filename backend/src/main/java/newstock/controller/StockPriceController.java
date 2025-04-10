package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.controller.response.StockPriceResponse;
import newstock.domain.stockprice.service.StockPriceInfoService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/v1/stockprice")
@RequiredArgsConstructor
public class StockPriceController {

    private final StockPriceInfoService stockPriceInfoService;

    @GetMapping("/{stockId}")
    public ResponseEntity<Api<StockPriceResponse>> getAllStockPricesByStockId(@PathVariable Integer stockId) {

        StockPriceResponse stockPriceResponse = stockPriceInfoService.getAllStockPrices(stockId);
        return ResponseEntity.ok(Api.ok(stockPriceResponse));
    }

}