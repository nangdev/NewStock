FROM gradle:7.4.2-jdk17 AS build
WORKDIR /app
COPY . .
RUN gradle build --no-daemon

FROM eclipse-temurin:17.0.12_7-jre-jammy
COPY --from=build /app/build/libs/*.jar app.jar
COPY .env /.env
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app.jar"]