package newstock.external.kis;

import lombok.Data;

@Data
public class KisWebSocketSubMsg {
    private KisHeaderDto header;
    private KisBodyDto body;
}
