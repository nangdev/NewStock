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

export type KakaoLoginReqType = {
  code: string;
  fcmToken: string;
};

export type KakaoLoginResType = BaseResType & {
  data: {
    accessToken: string;
    refreshToken: string;
  };
};

export type LogoutResType = BaseResType;
