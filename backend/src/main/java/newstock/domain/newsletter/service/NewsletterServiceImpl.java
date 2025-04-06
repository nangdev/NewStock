package newstock.domain.newsletter.service;

import lombok.RequiredArgsConstructor;
import newstock.controller.request.NewsletterContentRequest;
import newstock.domain.keyword.dto.KeywordItem;
import newstock.domain.keyword.dto.KeywordRequest;
import newstock.domain.keyword.dto.KeywordResponse;
import newstock.domain.keyword.service.KeywordService;
import newstock.domain.news.entity.News;
import newstock.domain.news.repository.NewsRepository;
import newstock.domain.newsletter.dto.NewsletterDto;
import newstock.controller.request.NewsletterRequest;
import newstock.controller.response.NewsletterResponse;
import newstock.domain.newsletter.entity.Newsletter;
import newstock.domain.newsletter.repository.NewsletterRepository;
import newstock.domain.stock.service.StockService;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import newstock.external.chatgpt.ChatGPTClient;
import newstock.external.chatgpt.dto.ChatGPTRequest;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
@RequiredArgsConstructor
public class NewsletterServiceImpl implements NewsletterService {

    private final ChatGPTClient chatGPTClient;

    private final KeywordService keywordService;

    private final NewsletterRepository newsletterRepository;

    private final NewsRepository newsRepository;

    private final StockService stockService;

    @Override
    public NewsletterResponse getNewsletterByDate(NewsletterRequest newsletterRequest) {

        List<NewsletterDto> newsletterDtoList = new ArrayList<>();

        for(Integer stockId: newsletterRequest.getStockIdList()){

            Newsletter newsletter = newsletterRepository.findByStockIdAndDate(stockId,newsletterRequest.getDate())
                    .orElseThrow(()-> new DbException(ExceptionCode.NEWS_LETTER_NOT_FOUND));
            KeywordResponse keywordList = keywordService.getKeywordsByStockId(KeywordRequest.of(stockId,newsletterRequest.getDate()));
            List<KeywordItem> keywordItems = keywordList.getKeywords();
            newsletterDtoList.add(NewsletterDto.of(stockId, newsletter.getContent(), keywordItems));

        }

        return NewsletterResponse.of(newsletterDtoList);
    }

    @Override
    public void addNewsletter(Integer stockId) {

            List<News> newsList = newsRepository.getTopNewsListByStockId(stockId)
                    .orElseThrow(() -> new DbException(ExceptionCode.NEWS_NOT_FOUND));

            StringBuilder combinedContent = new StringBuilder();
            for (News news : newsList) {
                combinedContent.append(news.getContent()).append("\n");
            }

            String prompt = "주어진 기사들을 각각 한 줄로 요약해 주세요. 만약 기사 수가 5개 미만이면, 요약 결과는 주어진 기사 수 만큼만 출력해 주세요. 각 요약문에서 핵심 키워드를 **굵게 강조**하고, 다른 말은 하지말고 결과만 마크다운 형식으로 제공해 주세요.\n\n"
                    + combinedContent;

            ChatGPTRequest.Message message = new ChatGPTRequest.Message("user", prompt);

            ChatGPTRequest chatRequest = ChatGPTRequest.builder()
                    .model("gpt-4o-mini")
                    .messages(Collections.singletonList(message))
                    .temperature(0.2)
                    .build();

            chatGPTClient.sendChatRequest(chatRequest)
                    .subscribe(response -> {
                        String summarizedContent = response.getChoices().get(0).getMessage().getContent();

                        String formattedDate = LocalDate.now().format(DateTimeFormatter.ofPattern("yyMMdd"));

                        Newsletter newsletter = Newsletter.builder()
                                .stockId(stockId)
                                .content(summarizedContent)
                                .date(formattedDate)
                                .build();
                        newsletterRepository.save(newsletter);
                    }, error -> {
                        // 에러 처리
                        System.err.println("ChatGPT 요청 중 오류 발생: " + error.getMessage());
                    });
    }

    @Override
    public void addNewsletterByContent(Integer stockId, NewsletterContentRequest content) {

        StringBuilder combinedContent = new StringBuilder();
        for (String text : content.getContents()) {
            combinedContent.append(text).append("\n");
        }

        String prompt = "주어진 기사들을 각각 한 줄로 요약해 주세요. 만약 기사 수가 5개 미만이면, 요약 결과는 주어진 기사 수 만큼만 출력해 주세요. 각 요약문에서 핵심 키워드를 **굵게 강조**하고, 다른 말은 하지말고 결과만 마크다운 형식으로 제공해 주세요.\n\n\n"
                + combinedContent;

        ChatGPTRequest.Message message = new ChatGPTRequest.Message("user", prompt);

        ChatGPTRequest chatRequest = ChatGPTRequest.builder()
                .model("gpt-4o-mini")
                .messages(Collections.singletonList(message))
                .temperature(0.2)
                .build();

        chatGPTClient.sendChatRequest(chatRequest)
                .subscribe(response -> {
                    String summarizedContent = response.getChoices().get(0).getMessage().getContent();

                    String formattedDate = LocalDate.now().format(DateTimeFormatter.ofPattern("yyMMdd"));

                    Newsletter newsletter = Newsletter.builder()
                            .stockId(stockId)
                            .content(summarizedContent)
                            .date(formattedDate)
                            .build();
                    newsletterRepository.save(newsletter);
                }, error -> {
                    // 에러 처리
                    System.err.println("ChatGPT 요청 중 오류 발생: " + error.getMessage());
                });
    }

}
