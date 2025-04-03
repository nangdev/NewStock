package newstock.domain.keyword.repository;

import newstock.domain.keyword.entity.Keyword;

import java.util.List;

public interface KeywordCustomRepository {

    List<Keyword> findByStockIdAndDate(Integer stockId, String date);
}
