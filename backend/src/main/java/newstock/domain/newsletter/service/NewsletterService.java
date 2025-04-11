package newstock.domain.newsletter.service;

import newstock.controller.request.NewsletterContentRequest;
import newstock.controller.request.NewsletterRequest;
import newstock.controller.response.NewsletterResponse;

public interface NewsletterService {

    NewsletterResponse getNewsletterByDate(NewsletterRequest newsletterRequest);

    void addNewsletter(Integer stockId);

    void addNewsletterByContent(Integer stockId, NewsletterContentRequest content);

    void addNewsletterAndKeyword(String date);
}
