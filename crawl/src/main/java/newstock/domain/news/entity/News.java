package newstock.domain.news.entity;

import jakarta.persistence.*;
import lombok.*;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name="news")
public class News {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    int id;

    String stockCode;

    String title;

    String description;

    @Lob
    String content;

    String newsImage;

    String url;

    String press;

    String pressLogo;

    String publishedDate;

    String newsSummary;

    String score;

    String keyword;

    public static News of(newstock.domain.news.dto.NewsItem dto) {
        return News.builder()
                .stockCode(dto.getStockCode())
                .title(dto.getTitle())
                .description(dto.getDescription())
                .content(dto.getContent())
                .newsImage(dto.getNewsImage())
                .url(dto.getUrl())
                .press(dto.getPress())
                .pressLogo(dto.getPressLogo())
                .publishedDate(dto.getPublishedDate())
                // 추후 AI 처리 후 업데이트할 정보는 null로 설정
                .newsSummary(null)
                .score(null)
                .keyword(null)
                .build();
    }

}
