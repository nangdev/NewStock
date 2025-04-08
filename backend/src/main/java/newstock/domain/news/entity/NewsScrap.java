package newstock.domain.news.entity;

import jakarta.persistence.*;
import lombok.*;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name="news_scrap")
public class NewsScrap {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer scrapId;

    private Integer userId;

    private Integer newsId;

}
