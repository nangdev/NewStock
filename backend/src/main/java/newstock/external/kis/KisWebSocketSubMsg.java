package newstock.external.kis;

import lombok.Data;
import newstock.external.kis.dto.KisBodyDto;
import newstock.external.kis.dto.KisHeaderDto;

@Data
public class KisWebSocketSubMsg {
    private KisHeaderDto header;
    private KisBodyDto body;
}
