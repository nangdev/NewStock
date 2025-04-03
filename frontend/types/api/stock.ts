import { BaseResType } from './base';

export type AllStockResType = BaseResType & {
  data: {
    stockList: {
      stockId: number;
      stockCode: string;
      stockName: string;
      isInterested: boolean;
    }[];
  };
};

export type StockInterestReqType = {
  stockIdList: number[];
};

export type StockInterestResType = BaseResType;

export type AllUserStockResType = BaseResType & {
  data: {
    stockList: {
      stockId: number,
      stockCode: string,
      stockName: string,
      closingPrice: number,
      rcPdcp: number,
      imgUrl: string,
    }[];
  }
}

export type StockDetailInfoResType = BaseResType & {
  data: {
    stockId: number,
    stockName: string,
    closingPrice: number,
    rcPdcp: number,
    stockImage: string,
    totalPrice: string,
    capital: string,
    lstgStqt: string,
    parValue: string,
    issuePrice: string,
    listingDate: string,
    stdIccn: string,
  }
}