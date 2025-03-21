package newstock.domain.news.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.*;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
public class News {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    int id;

    String stockCode;

    String title;

    String description;

    String content;

    String newsImage;

    String url;

    String press;

    String pressLogo;

    String publishedDate;

    String newsSummary;

    String score;

    String keyword;

}
