package newstock.common.jwt;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.GenericFilterBean;
import newstock.common.jwt.JwtTokenProvider;

import java.io.IOException;

@Slf4j
@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends GenericFilterBean {

    private final JwtTokenProvider jwtTokenProvider;

    /**
     * 모든 요청마다 실행되는 필터 메서드
     */
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {

        // 헤더에서 JWT 토큰 추출
        String token = resolveToken((HttpServletRequest) request);

        // 유효한 토큰이라면 Authentication 객체를 꺼내서 SecurityContext에 등록
        if (StringUtils.hasText(token) && jwtTokenProvider.validateToken(token)) {
            Authentication authentication = jwtTokenProvider.getAuthentication(token); // 유저 정보 복원
            SecurityContextHolder.getContext().setAuthentication(authentication); // 인증 등록
            log.debug("JWT 인증 성공 - 사용자: {}", authentication.getName());
        }

        // 다음 필터로 요청 전달
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
