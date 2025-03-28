import { UserInfoType } from 'types/user';

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

export type EmailCheckReqType = {
  email: string;
};

export type EmailCheckResType = BaseResType & {
  data: {
    isDuplicated: boolean;
  };
};

export type UserInfoResType = BaseResType & {
  data: UserInfoType;
};
