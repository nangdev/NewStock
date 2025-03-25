package newstock.common.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import newstock.domain.stock.service.StockPriceService;
import newstock.external.kis.KisOAuthClient;
import newstock.external.kis.KisWebSocketClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {
        registry.setApplicationDestinationPrefixes("/app");
        registry.enableSimpleBroker("/topic");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws")
                .setAllowedOriginPatterns("*")
                .withSockJS();
    }

    @Bean
    public KisWebSocketClient kisWebSocketClient(
            KisOAuthClient authClient, ObjectMapper objectMapper, StockPriceService stockPriceService
    ) throws Exception {
        return new KisWebSocketClient(authClient, objectMapper, stockPriceService);
    }
}
