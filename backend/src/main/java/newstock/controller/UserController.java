package newstock.controller;

import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import newstock.controller.response.UserResponse;
import newstock.domain.user.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/users")
public class UserController {

    private final UserService userService;

    /**
     * ID로 User 조회
     * @param id 조회할 User의 ID
     * @return User 정보
     */
    @GetMapping("/{id}")
    @Operation(summary = "ID로 사용자 조회", description = "고유 ID를 사용하여 사용자의 상세 정보를 조회합니다.")
    public ResponseEntity<UserResponse> getUserById(@PathVariable int id) {

        UserResponse userResponse = userService.getUserById(id);

        return ResponseEntity.ok(userResponse);
    }

}
