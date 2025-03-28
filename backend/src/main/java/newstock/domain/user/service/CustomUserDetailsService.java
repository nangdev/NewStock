package newstock.domain.user.service;

import lombok.RequiredArgsConstructor;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserRepository;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;

    /**
     * Spring Security가 내부적으로 호출하는 메서드
     * - 로그인 시 입력된 이메일(username)을 받아서
     * - 해당 유저를 DB에서 조회한 후
     * - CustomUserDetails로 감싸서 리턴함
     */
    @Override
    public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new UsernameNotFoundException("해당 이메일이 존재하지 않습니다."+ email));

        return new CustomUserDetails(user);
    }
}
