package newstock.domain.stock.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
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
@Data
public class StockDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

    private String stockImage;

    private boolean isInterested;

    public static StockDto of(Stock stock) {
        return StockDto.builder()
                .stockId(stock.getStockId())
                .stockCode(stock.getStockCode())
                .stockName(stock.getStockName())
                .stockImage(getBase64Image(stock.getImgUrl()))
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
