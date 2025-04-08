package newstock.domain.news.service;

import lombok.extern.slf4j.Slf4j;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.springframework.stereotype.Service;
import io.github.bonigarcia.wdm.WebDriverManager;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.net.MalformedURLException;
import java.net.URL;
import java.time.Duration;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

@Service
@Slf4j
public class WebDriverPoolService {

    private final int poolSize = 5;

    private final BlockingQueue<WebDriver> driverPool;
    private final AtomicInteger createdDrivers = new AtomicInteger(0);
    private volatile boolean shuttingDown = false;

    public WebDriverPoolService() {
        this.driverPool = new ArrayBlockingQueue<>(5);
    }

    @PostConstruct
    public void initialize() {
        log.info("WebDriver 풀 초기화 시작. 풀 크기: {}", poolSize);

        for (int i = 0; i < poolSize; i++) {
            try {
                WebDriver driver = createDriver();
                if (driver != null) {
                    driverPool.offer(driver);
                    log.info("WebDriver 풀에 드라이버 추가됨. 현재 풀 크기: {}", driverPool.size());
                }
            } catch (Exception e) {
                log.error("초기 WebDriver 생성 실패: {}", e.getMessage(), e);
            }
        }
    }

    public WebDriver getDriver() {
        if (shuttingDown) {
            throw new IllegalStateException("WebDriverPoolService가 종료 중입니다.");
        }

        WebDriver driver = null;
        try {
            int maxWaitSeconds = 30;
            driver = driverPool.poll(maxWaitSeconds, TimeUnit.SECONDS);

            // 풀에서 가져온 드라이버가 없거나 풀 사이즈가 최대치보다 작으면 새로 생성
            if (driver == null && createdDrivers.get() < poolSize) {
                log.info("풀에서 사용 가능한 WebDriver가 없습니다. 새 드라이버 생성 시도...");
                driver = createDriver();
                if (driver != null) {
                    createdDrivers.incrementAndGet();
                    log.info("새 WebDriver 생성 성공. 총 생성된 드라이버 수: {}", createdDrivers.get());
                }
            }
            
            // 드라이버 상태 확인
            if (driver != null) {
                try {
                    driver.getCurrentUrl(); // 드라이버 상태 확인
                } catch (Exception e) {
                    log.warn("손상된 WebDriver 감지. 새 드라이버 생성 시도...");
                    closeDriver(driver);
                    driver = createDriver();
                    if (driver != null) {
                        log.info("손상된 WebDriver 대체 성공");
                    }
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("WebDriver 획득 중 인터럽트 발생: {}", e.getMessage());
        } catch (Exception e) {
            log.error("WebDriver 획득 중 오류 발생: {}", e.getMessage(), e);
        }
        
        return driver;
    }

    public void releaseDriver(WebDriver driver) {
        if (driver == null) {
            return;
        }
        
        if (shuttingDown) {
            closeDriver(driver);
            return;
        }

        try {
            // 드라이버 상태 확인
            driver.getCurrentUrl();
            
            // 쿠키, 캐시 등 초기화
            driver.manage().deleteAllCookies();
            
            // 드라이버를 풀에 반환
            boolean added = driverPool.offer(driver);
            if (!added) {
                log.warn("WebDriver 풀이 가득 찼습니다. 드라이버를 종료합니다.");
                closeDriver(driver);
            }
        } catch (Exception e) {
            log.warn("손상된 WebDriver 감지. 드라이버를 종료합니다: {}", e.getMessage());
            closeDriver(driver);
            // 풀에 드라이버가 부족하면 새로 생성
            if (driverPool.size() < poolSize / 2) {
                try {
                    WebDriver newDriver = createDriver();
                    if (newDriver != null) {
                        driverPool.offer(newDriver);
                        log.info("손상된 WebDriver 대체. 현재 풀 크기: {}", driverPool.size());
                    }
                } catch (Exception ex) {
                    log.error("대체 WebDriver 생성 실패: {}", ex.getMessage());
                }
            }
        }
    }

    private WebDriver createDriver() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments(
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--window-size=1920,1080",
                "--disable-popup-blocking",
                "--blink-settings=imagesEnabled=false"
        );

        try {
            String remoteUrl = "http://selenium-hub:4444/wd/hub";
            RemoteWebDriver driver = new RemoteWebDriver(new URL(remoteUrl), options);

            // 타임아웃 설정
            driver.manage().timeouts().pageLoadTimeout(Duration.ofSeconds(30));
            driver.manage().timeouts().scriptTimeout(Duration.ofSeconds(20));
            driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));

            return driver;
        } catch (MalformedURLException e) {
            log.error("RemoteWebDriver URL 형식 오류: {}", e.getMessage(), e);
        } catch (Exception e) {
            log.error("RemoteWebDriver 초기화 실패: {}", e.getMessage(), e);
        }

    // 원격 드라이버 실패 시 로컬 드라이버 시도
    try {
        log.info("로컬 ChromeDriver로 대체 시도");
        WebDriverManager.chromedriver().setup();
        ChromeDriver driver = new ChromeDriver(options);
        
        // 타임아웃 설정 적용
        driver.manage().timeouts().pageLoadTimeout(Duration.ofSeconds(30));
        driver.manage().timeouts().scriptTimeout(Duration.ofSeconds(20));
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
        
        return driver;
    } catch (Exception e) {
        log.error("로컬 ChromeDriver 초기화 실패: {}", e.getMessage(), e);
        return null;
    }
    }

    private void closeDriver(WebDriver driver) {
        if (driver != null) {
            try {
                driver.quit();
                createdDrivers.decrementAndGet();
            } catch (Exception e) {
                log.error("WebDriver 종료 중 오류 발생: {}", e.getMessage(), e);
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        log.info("WebDriverPoolService 종료 중...");
        shuttingDown = true;
        
        // 풀에 있는 모든 드라이버 종료
        WebDriver driver;
        while ((driver = driverPool.poll()) != null) {
            closeDriver(driver);
        }
        
        log.info("WebDriverPoolService 종료 완료. 종료된 드라이버 수: {}", createdDrivers.get());
    }
}