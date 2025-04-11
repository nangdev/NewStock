package newstock.domain.keyword.service;

import newstock.domain.keyword.dto.KeywordDto;

import java.util.List;

public interface KeywordService {

    void addKeywords(List<KeywordDto> keywords);
}
