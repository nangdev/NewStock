package newstock.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

/**
 * HTML 페이지 렌더링을 위한 컨트롤러
 */
@Controller
public class PageController {

    @Value("${kakao.redirect-url}")
    private String kakaoRedirectUrl;

    /**
     * 카카오 인가 코드 받은 후 앱으로 딥링크 리디렉션할 HTML 페이지
     */
    @GetMapping("/kakao-redirect.html")
    public String kakaoRedirectPage(Model model) {
        model.addAttribute("kakaoRedirectUrl", kakaoRedirectUrl);
        return "kakao-redirect"; // templates/kakao-redirect.html
    }
}
