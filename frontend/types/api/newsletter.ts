import { BaseResType } from './base';

export type NewsletterReqType = {
  date: string;
};

export type NewsletterResType = BaseResType & {
  data: {
    newsletterList: {
      stockId: number;
      content: string;
      keywordList: {
        keyword: string;
        count: number;
      }[];
    }[];
  };
};
