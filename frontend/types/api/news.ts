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

export type NewsDetailResType = BaseResType & {
  data: {
    newsInfo: {
      title: string,
      content: string,
      newsImage: string,
      url: string,
      press: string,
      pressLogo: string,
      publishedDate: string,
      newsSummary: string,
      score: number,
    },
    isScraped: boolean,
  }
}