import { UserInfoType } from 'types/user';

import { BaseResType } from './base';

export type SignInReqType = {
  email: string;
  password: string;
  userName: string;
  nickname: string;
};

export type SignInResType = BaseResType;

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
