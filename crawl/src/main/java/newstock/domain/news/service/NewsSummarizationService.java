package newstock.domain.news.service;

public interface NewsSummarizationService {

    String summarize(String newsText, int maxLength, int minLength, boolean doSample);
}
