package newstock.external.kakao.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class KakaoUserInfo {
    private Long id; // 카카오 내부 유저 ID

    @JsonProperty("kakao_account")
    private KakaoAccount kakaoAccount;

    @Data
    public static class KakaoAccount {
        private String email;    // 유저 이메일
        private Profile profile; // 닉네임만 포함됨

        @Data
        public static class Profile {
            private String nickname; // 유저 닉네임
        }
    }
}
