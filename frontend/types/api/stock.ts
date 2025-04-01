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