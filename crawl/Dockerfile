FROM gradle:8.5-jdk17 AS build
WORKDIR /app
COPY . .
RUN gradle clean build -x test --no-daemon --refresh-dependencies --stacktrace

FROM eclipse-temurin:17.0.12_7-jre-jammy
RUN apt-get update && apt-get install -y wget gnupg2 \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && apt-get clean

COPY --from=build /app/build/libs/*.jar app.jar
COPY .env /.env
EXPOSE 8081
ENTRYPOINT ["java", "-jar", "/app.jar"]