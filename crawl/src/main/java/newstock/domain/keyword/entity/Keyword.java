package newstock.domain.keyword.entity;

import jakarta.persistence.*;
import lombok.*;
import newstock.domain.keyword.dto.KeywordDto;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name="news")
public class Keyword {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer keywordId;

    private Integer stockId;

    private String content;

    private String date;

    private int count;

    public static Keyword of(KeywordDto keywordDto) {
        return Keyword.builder()
                .stockId(keywordDto.getStockId())
                .content(keywordDto.getContent())
                .date(keywordDto.getDate())
                .count(keywordDto.getCount())
                .build();
    }
}
