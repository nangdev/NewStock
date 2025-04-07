import { BaseResType } from './base';

export type AllStockNewsResType = BaseResType & {
  data: {
    totalPage: number;
    newsList: {
      newsId: number,
      title: string,
      score: number,
      publishedDate: string, 
    }[],
  }
}

export type topFiveStockNewsResType = BaseResType & {
  data: {
    newsList: {
      newsId: number,
      title: string,
      score: number,
      publishedDate: string, 
    }[],
  }
}