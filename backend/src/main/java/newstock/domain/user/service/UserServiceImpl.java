package newstock.domain.user.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import newstock.controller.response.UserResponse;
import newstock.domain.user.entity.User;
import newstock.domain.user.repository.UserCustomRepository;
import newstock.domain.user.repository.UserRepository;
import newstock.exception.ExceptionCode;
import newstock.exception.type.DbException;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final UserCustomRepository userCustomRepository;

    @Override
    public UserResponse getUserById(int id) {


        // UserCustomRepository 사용 경우 ( QueryDsl )
        User user = userCustomRepository.findById(id)
                .orElseThrow(()-> new DbException(ExceptionCode.USER_NOT_FOUND));

        // UserRepository 사용 경우 ( JPA )
        User user2 = userRepository.findById(id)
                .orElseThrow(()-> new DbException(ExceptionCode.USER_NOT_FOUND));

        return UserResponse.of(user.getName(), user.getNickname(), user.getEmail());
    }
}
