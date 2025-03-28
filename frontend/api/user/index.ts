import { API_PATH } from 'constants/api';
import { EmailCheckReqType, EmailCheckResType, UserInfoResType } from 'types/api/auth';
import { SignInReqType, SignInResType } from 'types/api/user';

import { api } from '../axiosInstance';

export const postSignIn = async ({ email, password, userName, nickname }: SignInReqType) => {
  const response = await api.post<SignInResType>(API_PATH.USER.SIGN_IN, {
    email,
    password,
    userName,
    nickname,
  });
  return response.data;
};

export const getCheckEmail = async ({ email }: EmailCheckReqType) => {
  const response = await api.get<EmailCheckResType>(API_PATH.USER.CHECK_EMAIL, {
    params: { email },
  });
  return response.data;
};

export const getUserInfo = async () => {
  const response = await api.get<UserInfoResType>(API_PATH.USER.INFO);
  return response.data;
};
