package newstock.domain.stock.dto;

import lombok.Builder;
import lombok.Getter;
import newstock.domain.stock.entity.Stock;

import java.io.File;

@Builder
@Getter
public class StockInfoDto {

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;

    private File stockImage;

    private Integer totalPrice;

    private String capital;

    private String lstgStqt;

    private String parValue;

    private String issuePrice;

    private String listingDate;

    private String stdIccn;

    public static StockInfoDto of(Stock stock) {
        return StockInfoDto.builder()
                .stockName(stock.getStockName())
                .closingPrice(stock.getClosingPrice())
                .rcPdcp(stock.getRcPdcp())
                .stockImage(new File(stock.getImgUrl()))
                .totalPrice(stock.getTotalPrice())
                .capital(stock.getCapital())
                .listingDate(stock.getListingDate())
                .stdIccn(stock.getStdIccn())
                .lstgStqt(stock.getLstgStqt())
                .build();
    }

}
