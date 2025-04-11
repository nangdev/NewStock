package newstock.external.chatgpt;

import lombok.RequiredArgsConstructor;
import newstock.external.chatgpt.dto.ChatGPTRequest;
import newstock.external.chatgpt.dto.ChatGPTResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
@RequiredArgsConstructor
public class ChatGPTClient {

    private final WebClient webClient;

    @Value("${openai.api.key}")
    private String apiKey;

    public Mono<ChatGPTResponse> sendChatRequest(ChatGPTRequest request) {
        return webClient.post()
                .uri("https://api.openai.com/v1/chat/completions")
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(ChatGPTResponse.class);
    }
}
