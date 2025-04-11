package newstock.domain.stock.repository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Repository
@RequiredArgsConstructor
@Slf4j
public class JdbcStockRepository {

    private final JdbcTemplate jdbcTemplate;

    /**
     * 주식 정보를 일괄 업데이트하는 메서드
     * batchSize 단위로 나누어 대량 업데이트를 효율적으로 처리
     *
     * @param updateParams 업데이트할 파라미터 목록
     * @return 업데이트 된 행 수
     */
    @Transactional
    public int[] batchUpdateStockInfo(List<Object[]> updateParams) {
        final int batchSize = 50;
        final String sql = "UPDATE stock SET closing_price = ?, rc_pdcp = ?, total_price = ?, ctpd_price = ? WHERE stock_id = ?";

        List<int[]> updateCounts = new ArrayList<>();

        for (int i = 0; i < updateParams.size(); i += batchSize) {
            int endIndex = Math.min(i + batchSize, updateParams.size());
            List<Object[]> batch = updateParams.subList(i, endIndex);

            int[] counts = jdbcTemplate.batchUpdate(sql, batch);
            updateCounts.add(counts);

            log.info("배치 업데이트 진행: {} / {} 완료", endIndex, updateParams.size());
        }

        // 모든 배치 결과를 하나의 배열로 합침
        int totalUpdated = updateCounts.stream()
                .flatMapToInt(array -> java.util.Arrays.stream(array))
                .sum();

        log.info("총 {}개 종목 정보 업데이트 완료", totalUpdated);

        // 마지막 배치 결과 반환
        return updateCounts.isEmpty() ? new int[0] : updateCounts.get(updateCounts.size() - 1);
    }
} 