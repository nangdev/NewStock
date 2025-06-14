package newstock.domain.stock.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.external.kis.dto.KisRealTimeStockPriceDto;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class StockPriceServiceImpl implements StockPriceService {
    private final SimpMessagingTemplate messagingTemplate;

    @Override
    public void sendMessage(String message) {
        messagingTemplate.convertAndSend("/topic/rtp", message);
    }

    @Override
    public void sendStockInfo(KisRealTimeStockPriceDto RTStockInfoDto) {
        try {
            String stockCode = RTStockInfoDto.getStockCode();
            messagingTemplate.convertAndSend("/topic/rtp/" + stockCode, RTStockInfoDto);
        } catch (Exception e) {
            log.error("STOMP 메시지 전송 실패");
        }
    }
}
