package newstock.domain.keyword.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.keyword.dto.KeywordDto;
import newstock.domain.keyword.reposiory.KeywordRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class KeywordServiceImpl implements KeywordService {

    private final KeywordRepository keywordRepository;

    @Transactional
    @Override
    public void addKeywords(List<KeywordDto> keywords) {

    }
}
