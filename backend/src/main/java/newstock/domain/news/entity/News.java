package newstock.domain.news.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name="news")
public class News {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer newsId;

    private Integer stockId;

    private String title;

    private String description;

    @Lob
    private String content;

    private String newsImage;

    private String url;

    private String press;

    private String pressLogo;

    private String publishedDate;

    @Lob
    private String newsSummary;

    private int score;

}
