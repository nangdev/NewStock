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
        word: string;
        count: number;
      }[];
    }[];
  };
};
