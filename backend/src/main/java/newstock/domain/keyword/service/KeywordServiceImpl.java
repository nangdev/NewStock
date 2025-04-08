package newstock.domain.keyword.service;

import lombok.RequiredArgsConstructor;
import newstock.controller.request.NewsletterContentRequest;
import newstock.domain.keyword.dto.*;
import newstock.domain.keyword.entity.Keyword;
import newstock.domain.keyword.repository.KeywordRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class KeywordServiceImpl implements KeywordService {

    private final WebClient webClient;

    private final KeywordRepository keywordRepository;

    @Value("${news.ai.url}")
    private String newsAiUrl;

    @Override
    public KeywordAIResponse extractKeywords(KeywordAIRequest keywordAIRequest) {

        return webClient.post()
                .uri(newsAiUrl + "/keywords")
                .bodyValue(keywordAIRequest)
                .retrieve()
                .bodyToMono(KeywordAIResponse.class)
                .block();
    }

    @Transactional
    @Override
    public void addKeyword(KeywordList keywordList) {

        List<Keyword> keywords = keywordList.getKeywords().stream()
                .map(item -> Keyword.builder()
                        .stockId(keywordList.getStockId())
                        .content(item.getWord())
                        .count(item.getCount())
                        .date(java.time.LocalDate.now(ZoneId.of("Asia/Seoul")).format(DateTimeFormatter.ofPattern("yyyyMMdd")))
                        .build())
                .collect(Collectors.toList());

        keywordRepository.saveAll(keywords);
    }

    @Override
    public void addKeywordByContent(Integer stockId, NewsletterContentRequest newsletterContentRequest) {
        List<Article> articles = newsletterContentRequest.getContents().stream()
                .map(content -> Article.builder().content(content).build())
                .collect(Collectors.toList());
        KeywordAIResponse keywordAIResponse = extractKeywords(KeywordAIRequest.of(articles));

        KeywordList keywordList = KeywordList.of(keywordAIResponse.getKeywords(),stockId);

        List<Keyword> keywords = keywordList.getKeywords().stream()
                .map(item -> Keyword.builder()
                        .stockId(keywordList.getStockId())
                        .content(item.getWord())
                        .count(item.getCount())
                        .date(java.time.LocalDate.now(ZoneId.of("Asia/Seoul")).format(DateTimeFormatter.ofPattern("yyMMdd")))
                        .build())
                .collect(Collectors.toList());

        keywordRepository.saveAll(keywords);
    }

    @Override
    public KeywordResponse getKeywordsByStockId(KeywordRequest keywordRequest) {

        List<Keyword> keywordList = keywordRepository.findByStockIdAndDate(
                keywordRequest.getStockId(), keywordRequest.getDate());

        List<KeywordItem> keywordItems = keywordList.stream()
                .map(k -> KeywordItem.of(k.getContent(), k.getCount()))
                .collect(Collectors.toList());

        return KeywordResponse.of(keywordItems);
    }

}
