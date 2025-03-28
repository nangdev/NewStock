package newstock.common.jwt;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.redis.TokenBlacklistService;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.GenericFilterBean;

import java.io.IOException;

@Slf4j
@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends GenericFilterBean {

    private final JwtTokenProvider jwtTokenProvider;
    private final TokenBlacklistService tokenBlacklistService;

    /**
     * 모든 요청마다 실행되는 필터 메서드
     */
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {

        String token = resolveToken((HttpServletRequest) request);

        if (StringUtils.hasText(token) && jwtTokenProvider.validateToken(token)) {

            // 블랙리스트 확인
            if (tokenBlacklistService.isBlacklisted(token)) {
                log.warn("블랙리스트에 등록된 토큰입니다: {}", token);
                chain.doFilter(request, response);
                return;
            }

            // 정상 토큰
            Authentication authentication = jwtTokenProvider.getAuthentication(token);
            SecurityContextHolder.getContext().setAuthentication(authentication);
            log.debug("JWT 인증 성공 - 사용자: {}", authentication.getName());
        }

        chain.doFilter(request, response);
    }

    /**
     * Request Header에서 JWT 토큰 추출
     */
    private String resolveToken(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization"); // Request Header 중 Authorization 가져옴
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("Bearer ")) { // 빈 문자열이 아니고, "Bearer "로 시작하는지 확인
            return bearerToken.substring(7); // "Bearer "(총 7글자) 제거 후 실제 토큰만 꺼냄
        }
        return null;
    }
}
