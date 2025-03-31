package newstock.common.jwt;

import lombok.RequiredArgsConstructor;
import newstock.common.jwt.JwtTokenProvider;
import newstock.common.redis.RedisUtil;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class TokenBlacklistService {

    private final RedisUtil redisUtil;
    private final JwtTokenProvider jwtTokenProvider;

    private static final String BLACKLIST_PREFIX = "blacklist:";

    public void addToBlacklist(String token) {
        long remainingMillis  = jwtTokenProvider.getTokenRemainingTime(token);
        long seconds  = remainingMillis  / 1000;

        redisUtil.set(BLACKLIST_PREFIX + token, "logout", seconds);
    }

    public boolean isBlacklisted(String token) {
        return redisUtil.hasKey(BLACKLIST_PREFIX + token);
    }
}
