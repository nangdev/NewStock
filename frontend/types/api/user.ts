import { BaseResType } from './base';

export type SignInReqType = {
  email: string;
  password: string;
  userName: string;
  nickname: string;
};

export type SignInResType = BaseResType;
