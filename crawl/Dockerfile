FROM gradle:8.5-jdk17 AS build
WORKDIR /app
COPY . .
RUN gradle clean build -x test --no-daemon --refresh-dependencies --stacktrace

FROM eclipse-temurin:17.0.12_7-jre-jammy
# Selenium 의존성 설치
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    default-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/build/libs/*.jar app.jar
COPY .env /.env
EXPOSE 8081
ENTRYPOINT ["java", "-jar", "/app.jar"]