FROM gradle:7.4.2-jdk11 AS build
WORKDIR /app
COPY . .
RUN gradle build --no-daemon

FROM openjdk:17-jre-slim
COPY --from=build /app/build/libs/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app.jar"]