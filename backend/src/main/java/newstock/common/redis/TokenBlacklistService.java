package newstock.common.redis;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class TokenBlacklistService {

    private final RedisUtil redisUtil;

    private static final String BLACKLIST_PREFIX = "blacklist:";

    public void addToBlacklist(String token, long expirationMillis) {
        long expirationMinutes = expirationMillis / 1000 / 60;
        redisUtil.set(BLACKLIST_PREFIX + token, "logout", expirationMinutes);
    }

    public boolean isBlacklisted(String token) {
        return redisUtil.hasKey(BLACKLIST_PREFIX + token);
    }
}
