import { BaseResType } from './base';

export type AllStockResType = BaseResType & {
  data: {
    stockList: {
      stockId: number;
      stockCode: string;
      stockName: string;
      isInterested: boolean;
      imgUrl: string;
    }[];
  };
};

export type StockInterestReqType = {
  stockIdList: number[];
};

export type StockInterestResType = BaseResType;

export type AllUserStockResType = BaseResType & {
  data: {
    stockList: StockType[];
  };
};

export type StockDetailInfoResType = BaseResType & {
  data: {
    stockId: number;
    stockName: string;
    closingPrice: number;
    rcPdcp: number;
    stockImage: string;
    totalPrice: string;
    capital: string;
    lstgStqt: string;
    parValue: string;
    issuePrice: string;
    listingDate: string;
    stdIccn: string;
    ctpdPrice: number;
  };
};

export type StockType = {
  stockId: number;
  stockCode: string;
  stockName: string;
  closingPrice: number;
  rcPdcp: number;
  imgUrl: string;
};
