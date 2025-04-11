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
import java.text.DecimalFormat;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Base64;

@Builder
@NoArgsConstructor
@AllArgsConstructor
@Getter
public class StockInfoDto {

    private Integer stockId;

    private String stockCode;

    private String stockName;

    private Integer closingPrice;

    private String rcPdcp;  // 전일 대비 등략률

    private String stockImage;

    private String totalPrice;  // 시가 총액

    private String capital; // 자본금

    private String lstgStqt; // 상장 주수

    private Integer ctpdPrice; // 전일 대비 차이금액

    private String listingDate; // 상장 일자

    private String stdIccn; // 업종 이름

    public static StockInfoDto of(Stock stock) {
        return StockInfoDto.builder()
                .stockId(stock.getStockId())
                .stockName(stock.getStockName())
                .stockCode(stock.getStockCode())
                .closingPrice(stock.getClosingPrice())
                .rcPdcp(stock.getRcPdcp())
                .stockImage(getBase64Image(stock.getImgUrl()))
                .totalPrice(getFormattedTotalPrice(stock.getTotalPrice()))
                .capital(getFormattedCapital(stock.getCapital()))
                .listingDate(getFormattedListingDate(stock.getListingDate()))
                .stdIccn(stock.getStdIccn())
                .lstgStqt("총 "+stock.getLstgStqt()+"주")
                .ctpdPrice(stock.getCtpdPrice())
                .build();
    }

    // 자본금 포맷팅 (원 단위로 표시되어 있음)
    private static String getFormattedCapital(String capital) {
        if (capital == null || capital.isEmpty()) {
            return "-";
        }

        try {
            long capitalValue = Long.parseLong(capital);
            DecimalFormat df = new DecimalFormat("#,###");

            // 자본금을 억 단위로 변환
            double billions = capitalValue / 100000000.0;
            return df.format(billions) + "억원";
        } catch (NumberFormatException e) {
            return capital;
        }
    }

    // 시가총액 포맷팅 (백만원 단위로 표시되어 있음)
    private static String getFormattedTotalPrice(Integer totalPrice) {
        if (totalPrice == null) {
            return "-";
        }

        DecimalFormat df = new DecimalFormat("#,###");

        if (totalPrice >= 10) {
            // 1조원 이상
            return df.format(totalPrice / 10000) + "조 " + df.format(totalPrice % 10000) + "억원";
        } else {
            // 1조원 미만
            return df.format(totalPrice) + "억원";
        }
    }

    private static String getFormattedListingDate(String listingDate) {
        if (listingDate == null || listingDate.length() != 8) {
            return "-";
        }

        try {
            LocalDate date = LocalDate.parse(listingDate, DateTimeFormatter.ofPattern("yyyyMMdd"));
            return date.format(DateTimeFormatter.ofPattern("yyyy년 MM월 dd일"));
        } catch (Exception e) {
            return listingDate;
        }
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
