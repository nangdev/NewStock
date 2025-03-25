package newstock.domain.stock.service;

import lombok.RequiredArgsConstructor;
import newstock.external.kis.KisStockInfoDto;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class StockPriceServiceImpl implements StockPriceService {
    private final SimpMessagingTemplate messagingTemplate;

    @Override
    public void sendMessage(String message) {
        messagingTemplate.convertAndSend("/topic/rtp", message);
    }

    @Override
    public void sendStockInfo(KisStockInfoDto stockInfoDto) {
        String stockCode = stockInfoDto.getStockCode();
        messagingTemplate.convertAndSend("/topic/rtp/"+stockCode, stockInfoDto);
    }
}
