package newstock.domain.crawl.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

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
