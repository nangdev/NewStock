package newstock.common.redis;

import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;

@Component
@RequiredArgsConstructor
public class RedisUtil {

    private final RedisTemplate<String, Object> redisTemplate;

    /**
     * Redis에 key-value 저장 (TTL 설정 포함)
     * @param key 저장할 키
     * @param value 저장할 값
     * @param minutes 유효 시간 (분 단위)
     */
    public void set(String key, Object value, long minutes) {
        redisTemplate.opsForValue().set(key, value, minutes, TimeUnit.MINUTES);
    }

    /**
     * key로부터 값 가져오기
     */
    public Object get(String key) {
        return redisTemplate.opsForValue().get(key);
    }

    /**
     * key 삭제
     */
    public boolean delete(String key) {
        return Boolean.TRUE.equals(redisTemplate.delete(key));
    }

    /**
     * key 존재 여부 확인
     */
    public boolean hasKey(String key) {
        return Boolean.TRUE.equals(redisTemplate.hasKey(key));
    }
}
