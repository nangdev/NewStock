package newstock.common.jwt;

import io.jsonwebtoken.*;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import lombok.extern.slf4j.Slf4j;
import newstock.domain.user.entity.User;
import newstock.domain.user.dto.JwtToken;
import newstock.domain.user.repository.UserRepository;
import newstock.domain.user.service.CustomUserDetails;
import newstock.exception.ExceptionCode;
import newstock.exception.type.ValidationException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;

import java.security.Key;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.stream.Collectors;

@Slf4j
@Component
public class JwtTokenProvider {

    private final Key key;
    private final UserRepository userRepository;



    private static final long ACCESS_TOKEN_EXPIRE_TIME = 1000L * 60 * 60* 24 * 14;      // 2주
    private static final long REFRESH_TOKEN_EXPIRE_TIME = 1000L * 60 * 60 * 24 * 30;    // 1달

    // JWT 시크릿 키 설정
    public JwtTokenProvider(
            @Value("${jwt.secret}") String secretKey,
            UserRepository userRepository
    ) {
        byte[] keyBytes = Decoders.BASE64.decode(secretKey);
        this.key = Keys.hmacShaKeyFor(keyBytes);
        this.userRepository = userRepository;
    }

    /**
     * Authentication 객체로부터 JWT 토큰을 생성
     */
    public JwtToken generateToken(Authentication authentication) {
        CustomUserDetails userDetails = (CustomUserDetails) authentication.getPrincipal();
        Integer userId = userDetails.getUser().getUserId();
        String authorities = authentication.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.joining(","));

        long now = System.currentTimeMillis();
        Date accessTokenExpiresIn = new Date(now + ACCESS_TOKEN_EXPIRE_TIME);

        String accessToken = Jwts.builder()
                .setSubject(String.valueOf(userId))
                .claim("auth", authorities)
                .setExpiration(accessTokenExpiresIn)
                .signWith(key, SignatureAlgorithm.HS256)
                .compact();

        String refreshToken = Jwts.builder()
                .setExpiration(new Date(now + REFRESH_TOKEN_EXPIRE_TIME))
                .signWith(key, SignatureAlgorithm.HS256)
                .compact();

        return JwtToken.builder()
                .grantType("Bearer")
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .build();
    }


    /**
     * JWT 토큰의 유효성 검사
     * 정상 / 만료 / 조작 검증
     */
    public boolean validateToken(String token) {
        try {
            Jwts.parserBuilder()
                    .setSigningKey(key)
                    .build()
                    .parseClaimsJws(token);
            return true;
        } catch (SecurityException | MalformedJwtException e) {
            log.warn("Invalid JWT Token", e);
        } catch (ExpiredJwtException e) {
            log.warn("Expired JWT Token", e);
        } catch (UnsupportedJwtException e) {
            log.warn("Unsupported JWT Token", e);
        } catch (IllegalArgumentException e) {
            log.warn("JWT claims string is empty.", e);
        }
        return false;
    }

    /**
     * JWT AccessToken으로부터 Authentication 추출
     */
    public Authentication getAuthentication(String accessToken) {
        Claims claims = parseClaims(accessToken);

        if (claims.get("auth") == null) {
            throw new ValidationException(ExceptionCode.TOKEN_INVALID);
        }

        Integer userId = Integer.parseInt(claims.getSubject());

        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("유저를 찾을 수 없습니다."));

        CustomUserDetails userDetails = new CustomUserDetails(user);

        Collection<? extends GrantedAuthority> authorities = Arrays.stream(
                        claims.get("auth").toString().split(","))
                .map(SimpleGrantedAuthority::new)
                .toList();

        return new UsernamePasswordAuthenticationToken(userDetails, "", authorities);
    }

    /**
     * JWT에서 Claims(내용) 꺼내기
     */
    private Claims parseClaims(String token) {
        try {
            return Jwts.parserBuilder()
                    .setSigningKey(key)
                    .build()
                    .parseClaimsJws(token)
                    .getBody();
        } catch (ExpiredJwtException e) {
            return e.getClaims();
        }
    }
}
