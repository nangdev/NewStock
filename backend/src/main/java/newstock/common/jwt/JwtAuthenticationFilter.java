package newstock.common.jwt;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.common.dto.Api;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
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

    private void setErrorResponse(HttpServletResponse response, ExceptionCode ec) throws IOException {
        response.setStatus(ec.getStatus().value());
        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");

        Api<Integer> api = Api.ERROR(ec.getMessage(), ec.getCode());
        String json = new ObjectMapper().writeValueAsString(api);

        response.getWriter().write(json);
    }

    /**
     * 모든 요청마다 실행되는 필터 메서드
     */
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {

        HttpServletRequest httpRequest = (HttpServletRequest) request;
        HttpServletResponse httpResponse = (HttpServletResponse) response;

        String path = httpRequest.getServletPath();
        String method = httpRequest.getMethod();

        // JWT 검사 생략할 경로들 조건문
        if ((path.equals("/v1/users") && method.equalsIgnoreCase("POST")) ||
                path.equals("/v1/users/check-email") ||
                path.equals("/v1/auth/refresh") ||
                path.equals("/v1/auth/login")) {
            chain.doFilter(request, response);
            return;
        }

        String token = resolveToken(httpRequest);
        log.debug("추출된 토큰: {}", token);

        if (!StringUtils.hasText(token)) {
            log.warn("토큰이 누락되었습니다.");
            setErrorResponse(httpResponse, ExceptionCode.TOKEN_MISSING);
            return;
        }

        try {
            jwtTokenProvider.validateAccessToken(token);

            // 블랙리스트 확인
            if (tokenBlacklistService.isBlacklisted(token)) {
                log.warn("블랙리스트에 등록된 토큰입니다: {}", token);
                setErrorResponse(httpResponse, ExceptionCode.TOKEN_INVALID);
                return;
            }

            // 정상 토큰
            Authentication authentication = jwtTokenProvider.getAuthentication(token);
            SecurityContextHolder.getContext().setAuthentication(authentication);
            log.debug("JWT 인증 성공 - 사용자: {}", authentication.getName());

            chain.doFilter(request, response);

        } catch (ValidationException e) {
            log.warn("JWT 인증 실패: {}", e.getMessage());
            setErrorResponse(httpResponse, e.getExceptionCode());
        }
    }

    /**
     * Request Header에서 JWT 토큰 추출
     */
    private String resolveToken(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }
}
