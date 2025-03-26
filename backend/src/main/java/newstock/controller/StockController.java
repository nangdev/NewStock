package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.common.dto.Api;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.service.StockService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/stock")
public class StockController {
    private final StockService stockService;

    @GetMapping
    public Api<List<StockDto>> getAllStocks() {
        return Api.ok(stockService.findAll());
    }
}
