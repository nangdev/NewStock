package newstock.domain.news.service;

import newstock.domain.keyword.dto.KeywordRequest;
import newstock.domain.news.dto.AnalysisRequest;
import newstock.domain.news.dto.AnalysisResponse;
import newstock.domain.keyword.dto.KeywordDto;
import newstock.domain.keyword.dto.KeywordResponse;
import newstock.domain.news.dto.SummarizationRequest;
import newstock.domain.news.dto.SummarizationResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@RequiredArgsConstructor
@Service
@Slf4j
public class NewsAiServiceImpl implements NewsAiService {

    @Value("${news.ai.url}")
    private String newsAiUrl;

    private final WebClient.Builder webClientBuilder;

    @Override
    public SummarizationResponse summarize(SummarizationRequest summarizationRequest) {
        WebClient webClient = webClientBuilder
                .baseUrl(newsAiUrl + "/summarize")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();

        Map<String, Object> requestData = new HashMap<>();
        requestData.put("content", summarizationRequest.getContent());
        requestData.put("max_length", summarizationRequest.getMaxLength());
        requestData.put("min_length", summarizationRequest.getMinLength());
        requestData.put("do_sample", summarizationRequest.isDoSample());

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
                .orElseThrow(() -> new RuntimeException("요약 응답 없음"));
    }


    @Override
    public AnalysisResponse analysis(AnalysisRequest analysisRequest) {
        WebClient webClient = webClientBuilder
                .baseUrl(newsAiUrl + "/score")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();

        Map<String, String> requestPayload = new HashMap<>();
        requestPayload.put("title", analysisRequest.getTitle());
        requestPayload.put("content", analysisRequest.getContent());

        AnalysisResponse response = webClient.post()
                .bodyValue(requestPayload)
                .retrieve()
                .onStatus(status -> !status.is2xxSuccessful(),
                        res -> res.bodyToMono(String.class)
                                .flatMap(err -> Mono.error(new RuntimeException("분석 API 호출 실패: " + err)))
                )
                .bodyToMono(AnalysisResponse.class)
                .block();

        return Optional.ofNullable(response)
                .orElseThrow(() -> new RuntimeException("분석 응답 없음"));
    }

    @Override
    public KeywordResponse getKeyword(KeywordRequest keywordRequest) {
        WebClient webClient = webClientBuilder
                .baseUrl(newsAiUrl + "/keyword")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();

        Map<String, String> requestData = new HashMap<>();
        requestData.put("content", keywordRequest.getNewsText());

        List<KeywordDto> keywords = webClient.post()
                .bodyValue(requestData)
                .retrieve()
                .onStatus(status -> !status.is2xxSuccessful(),
                        res -> res.bodyToMono(String.class)
                                .flatMap(err -> Mono.error(new RuntimeException("키워드 API 호출 실패: " + err)))
                )
                .bodyToMono(new ParameterizedTypeReference<List<KeywordDto>>() {})
                .block();

        // 오늘 날짜로 설정하고, 전달받은 stockId를 모두 적용
        String today = LocalDate.now().toString();
        List<KeywordDto> updatedKeywords = keywords.stream()
                .map(dto -> KeywordDto.builder()
                        .content(dto.getContent())
                        .stockId(keywordRequest.getStockId())
                        .date(today)
                        .count(dto.getCount())
                        .build())
                .collect(Collectors.toList());

        return Optional.of(updatedKeywords)
                .map(list -> KeywordResponse.builder().keywords(list).build())
                .orElseThrow(() -> new RuntimeException("키워드 응답 없음"));
    }
}

