package newstock.domain.newsletter.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.keyword.dto.KeywordItem;
import newstock.domain.keyword.dto.KeywordRequest;
import newstock.domain.keyword.dto.KeywordResponse;
import newstock.domain.keyword.entity.Keyword;
import newstock.domain.keyword.repository.KeywordRepository;
import newstock.domain.keyword.service.KeywordService;
import newstock.domain.newsletter.dto.NewsletterDto;
import newstock.domain.newsletter.dto.NewsletterRequest;
import newstock.domain.newsletter.dto.NewsletterResponse;
import newstock.domain.newsletter.entity.Newsletter;
import newstock.domain.newsletter.repository.NewsletterRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import newstock.external.chatgpt.ChatGPTClient;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class NewsletterServiceImpl implements NewsletterService {

    private final ChatGPTClient chatGPTClient;

    private final KeywordService keywordService;

    private final NewsletterRepository newsletterRepository;

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
    public void addNewsletter(NewsletterRequest newsletterRequest) {




    }
}
