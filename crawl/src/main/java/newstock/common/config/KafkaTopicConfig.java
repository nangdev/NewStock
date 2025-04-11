package newstock.common.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

@Configuration
public class KafkaTopicConfig {

    @Bean
    public NewTopic newsCrawlTopic() {
        return TopicBuilder.name("news-crawl-topic")
                .partitions(10)
                .replicas(1)
                .build();
    }

    @Bean
    public NewTopic newsAiTopic() {
        return TopicBuilder.name("news-ai-topic")
                .partitions(10)
                .replicas(1)
                .build();
    }

    @Bean
    public NewTopic newsDbTopic() {
        return TopicBuilder.name("news-db-topic")
                .partitions(10)
                .replicas(1)
                .build();
    }
}
