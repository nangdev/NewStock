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
