import { BaseResType } from './base';

export type AllStockNewsResType = BaseResType & {
  data: {
    totalPage: number;
    newsList: {
      newsId: number,
      title: string,
      description: string,
      score: number,
      publishedDate: string, 
    }[],
  }
}