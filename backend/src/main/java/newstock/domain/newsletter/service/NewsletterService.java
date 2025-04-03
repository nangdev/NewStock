package newstock.domain.newsletter.service;

import newstock.domain.newsletter.dto.NewsletterRequest;
import newstock.domain.newsletter.dto.NewsletterResponse;

public interface NewsletterService {

    NewsletterResponse getNewsletterByDate(NewsletterRequest newsletterRequest);

    void addNewsletter(NewsletterRequest newsletterRequest);
}
