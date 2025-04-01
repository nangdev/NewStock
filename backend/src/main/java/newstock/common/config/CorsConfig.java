package newstock.common.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**") // 모든 API 경로 허용
                .allowedOrigins("http://localhost:5500") // 프론트 주소 (운영 배포 시 변경)
                .allowedMethods("*") // GET, POST, PUT, DELETE 등 전부 허용
                .allowedHeaders("*") // 어떤 헤더든 허용 (Authorization 포함)
                .allowCredentials(true); // 인증 정보 포함 허용 (쿠키, Authorization 헤더 등)
    }
}
