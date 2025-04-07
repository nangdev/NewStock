package newstock.domain.stock.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.stock.dto.StockCodeDto;
import newstock.domain.stock.dto.StockDto;
import newstock.domain.stock.dto.StockInfoDto;
import newstock.domain.stock.dto.UserStockDto;
import newstock.domain.stock.entity.Stock;
import newstock.domain.stock.entity.UserStock;
import newstock.domain.stock.repository.JdbcStockRepository;
import newstock.domain.stock.repository.StockRepository;
import newstock.domain.stock.repository.UserStockRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import newstock.exception.type.InternalException;
import newstock.external.kis.KisOAuthClient;
import newstock.external.kis.dto.KisStockInfoDto;
import newstock.external.kis.response.KisAccessTokenResponse;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class StockService {

    private final UserStockRepository userStockRepository;
    private final StockRepository stockRepository;
    private final KisOAuthClient kisOAuthClient;
    private final JdbcStockRepository jdbcStockRepository;

    public List<StockDto> getStockList(Integer userId) {
        List<StockDto> stockDtoList = stockRepository.findAll()
                .stream()
                .map(StockDto::of)
                .toList();

        List<UserStock> userStockList = userStockRepository.findUserStocksByUserId(userId);

        try {
            for (UserStock userStock : userStockList) {
                stockDtoList.get(userStock.getStockId() - 1).setInterested(true);
            }
        }catch (Exception e){
            throw new InternalException(ExceptionCode.INTERNAL_SERVER_ERROR);
        }

        return stockDtoList;
    }

    public List<StockDto> getAllStockList() {

        return stockRepository.findAll()
                .stream()
                .map(StockDto::of)
                .toList();
    }

    public List<UserStockDto> getUserStockList(Integer userId) {
        List<UserStockDto> userStockDtoList = stockRepository.findUserStocksByUserId(userId);

        for (UserStockDto userStockDto : userStockDtoList) {
            String imgUrl = userStockDto.getImgUrl();

            userStockDto.setImgUrl(getBase64Image(imgUrl));
        }

        return userStockDtoList;
    }

    @Transactional
    public void updateUserStockList(Integer userId, List<Integer> stockIdList) {
        try {
            userStockRepository.deleteUserStocksByUserId(userId);

            for (Integer stockId : stockIdList) {
                userStockRepository.save(UserStock.of(userId, stockId));
            }

        } catch (Exception e) {
            throw new DbException(ExceptionCode.USER_STOCK_UPDATE_FAILED);
        }
    }

    public StockInfoDto getStockInfo(Integer stockId) {
        Stock stock = stockRepository.findById(stockId)
                .orElseThrow(() -> new DbException(ExceptionCode.STOCK_NOT_FOUND));

        return StockInfoDto.of(stock);
    }

    /**
     * 평일 오후 3시 31분에 한국투자증권 API를 통해 주식 정보를 조회합니다.
     * 장 마감 직후 최종 주식 정보를 가져옵니다.
     */
    @Transactional
    @Scheduled(cron = "0 31 15 * * MON-FRI")
    public void fetchDailyStockInfo() {
        log.info("오후 3시 31분 종목 당일 종가 조회 시작");

        try {
            KisAccessTokenResponse resp = kisOAuthClient.getAccessToken();
            String accessToken = resp.getAccessToken();

            log.info("accessToken: {}", accessToken);

            List<StockCodeDto> stockCodes = stockRepository.findAllStockCodes();

            List<Object[]> batchUpdateParams = new ArrayList<>();

            for (StockCodeDto dto : stockCodes) {
                try {
                    String stockCode = dto.getStockCode();
                    // API 호출
                    KisStockInfoDto stockInfo = kisOAuthClient.getStockInfo(stockCode, accessToken);

                    // 업데이트할 데이터 준비
                    Object[] params = new Object[5];
                    params[0] = Integer.parseInt(stockInfo.getClosingPrice());  // 종가
                    params[1] = stockInfo.getRcPdcp();                          // 등락률 
                    params[2] = Integer.parseInt(stockInfo.getTotalPrice());    // 시가총액
                    params[3] = Integer.parseInt(stockInfo.getCtpdPrice());     // 전일대비
                    params[4] = dto.getStockId();                               // WHERE 조건용 코드

                    batchUpdateParams.add(params);

                    log.info("주식 [{}] 정보 조회 완료: {}", stockCode, stockInfo.getClosingPrice());

                    // API 호출 제한 고려하여 약간의 지연 추가
                    Thread.sleep(100);
                } catch (Exception e) {
                    log.error("주식 [{}] 정보 조회 실패: {}", dto.getStockCode(), e.getMessage());
                }
            }

            // JDBC 템플릿을 사용하여 배치 업데이트 실행
            if (!batchUpdateParams.isEmpty()) {
                jdbcStockRepository.batchUpdateStockInfo(batchUpdateParams);
            }
        } catch (Exception e) {
            log.error("주식 정보 업데이트 중 오류 발생: {}", e.getMessage());
        }

        log.info("종목 당일 종가 조회 끝 시각 : {}", LocalDateTime.now());
    }

    private String getBase64Image(String imgUrl) {
        File imageFile = new File(imgUrl);

        byte[] fileContent;

        try {
            fileContent = Files.readAllBytes(imageFile.toPath());
        } catch (IOException e) {
            throw new InternalException(ExceptionCode.STOCK_IMAGE_CHANGE_FAIELD);
        }

        return Base64.getEncoder().encodeToString(fileContent);
    }

}
