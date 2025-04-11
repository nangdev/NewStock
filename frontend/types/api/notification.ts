import { BaseResType } from './base';

export type NotificationType = {
  unId: number;
  newsInfo: {
    newsId: number;
    title: string;
    publishedDate: string;
  };
  stockInfo: {
    stockId: number;
    stockCode: string;
    stockName: string;
  };
  isRead: boolean;
};

export type NotificationListResType = BaseResType & {
  data: {
    notificationList: NotificationType[];
  };
};

export type NotificationReadResType = BaseResType;

export type NotificationDeleteResType = BaseResType;
