package newstock.domain.stock.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import newstock.domain.stock.entity.Stock;
import newstock.exception.ExceptionCode;
import newstock.exception.type.InternalException;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Base64;

@Builder
@NoArgsConstructor
@AllArgsConstructor
@Getter
public class StockInfoDto {

    private Integer stockId;

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;

    private String stockImage;

    private Integer totalPrice;

    private String capital;

    private String lstgStqt;

    private String parValue;

    private String issuePrice;

    private String listingDate;

    private String stdIccn;

    public static StockInfoDto of(Stock stock) {
        return StockInfoDto.builder()
                .stockId(stock.getStockId())
                .stockName(stock.getStockName())
                .closingPrice(stock.getClosingPrice())
                .rcPdcp(stock.getRcPdcp())
                .stockImage(StockInfoDto.getBase64Image(stock.getImgUrl()))
                .totalPrice(stock.getTotalPrice())
                .capital(stock.getCapital())
                .listingDate(stock.getListingDate())
                .stdIccn(stock.getStdIccn())
                .lstgStqt(stock.getLstgStqt())
                .build();
    }

    private static String getBase64Image(String imgUrl) {
        File imageFile = new File(imgUrl);

        byte[] fileContent;

        try {
            fileContent = Files.readAllBytes(imageFile.toPath());
        } catch (IOException e) {
            throw new InternalException(ExceptionCode.STOCK_IMAGE_CHANGE_FAIELD);
        }

        return Base64.getEncoder().encodeToString(fileContent);
    }

}
