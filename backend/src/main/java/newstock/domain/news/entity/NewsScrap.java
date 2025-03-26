package newstock.domain.news.entity;

import jakarta.persistence.*;
import lombok.*;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name="news_scrap")
public class NewsScrap {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int scrapId;

    private int userId;

    private int newsId;

}
