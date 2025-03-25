package newstock.domain.news.service;

import newstock.domain.news.dto.SummarizationResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@RequiredArgsConstructor
@Service
@Slf4j
public class NewsSummarizationServiceImpl implements NewsSummarizationService {

    @Value("${news.summarization.url}")
    private String summarizationUrl;

    private final WebClient.Builder webClientBuilder;

    @Override
    public String summarize(String newsText, int maxLength, int minLength, boolean doSample) {
        WebClient webClient = webClientBuilder
                .baseUrl(summarizationUrl)
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();

        Map<String, Object> requestData = new HashMap<>();
        requestData.put("news_text", newsText);
        requestData.put("max_length", maxLength);
        requestData.put("min_length", minLength);
        requestData.put("do_sample", doSample);

        SummarizationResponse response = webClient.post()
                .bodyValue(requestData)
                .retrieve()
                .onStatus(status -> !status.is2xxSuccessful(),
                        res -> res.bodyToMono(String.class)
                                .flatMap(err -> Mono.error(new RuntimeException("요약 API 호출 실패: " + err)))
                )
                .bodyToMono(SummarizationResponse.class)
                .block();

        return Optional.ofNullable(response)
                .map(SummarizationResponse::getSummaryText)
                .orElseThrow(() -> new RuntimeException("요약 응답 없음"));
    }
}

