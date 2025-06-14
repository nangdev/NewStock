import { UserInfoType } from 'types/user';

import { BaseResType } from './base';

export type SignUpReqType = {
  email: string;
  password: string;
  nickname: string;
};

export type SignUpResType = BaseResType;

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

export type UserNicknameReqType = {
  nickname: string;
};

export type UserNicknameResType = BaseResType & {
  data: {
    nickname: string;
  };
};

export type VerifySendReqType = {
  email: string;
};

export type VerifySendResType = BaseResType;

export type VerifyCheckReqType = {
  email: string;
  authCode: string;
};

export type VerifyCheckResType = BaseResType;
