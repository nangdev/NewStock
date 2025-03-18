package newstock.domain.user.repository;

import newstock.domain.user.entity.User;

import java.util.Optional;

public interface UserCustomRepository {

    Optional<User> findById(int id);
}
