package newstock.controller;

import lombok.RequiredArgsConstructor;
import newstock.controller.response.KisAccessTokenResponse;
import newstock.external.kis.KisOAuthClient;
import newstock.external.kis.response.KisWebSocketKeyResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/v1/stock")
public class StockController {

    private final KisOAuthClient client;

    @GetMapping("/key")
    public ResponseEntity<KisWebSocketKeyResponse> getKey() {
        return ResponseEntity.status(HttpStatus.OK).body(client.getWebSocketKey());
    }

    @GetMapping("/token")
    public ResponseEntity<KisAccessTokenResponse> getToken() {
        return ResponseEntity.status(HttpStatus.OK).body(client.getAccessToken());
    }
}
