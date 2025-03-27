import { BaseResType } from './base';

export type LoginReqType = {
  email: string;
  password: string;
  fcmToken: string;
};

export type LoginResType = BaseResType & {
  data: {
    accessToken: string;
    refreshToken: string;
  };
};
