import { API_PATH } from 'constants/api';
import {
  KakaoLoginReqType,
  KakaoLoginResType,
  LoginReqType,
  LoginResType,
  LogoutResType,
} from 'types/api/auth';

import { api } from '../axiosInstance';

export const postLogin = async ({ email, password, fcmToken }: LoginReqType) => {
  const response = await api.post<LoginResType>(API_PATH.AUTH.LOGIN, { email, password, fcmToken });
  return response.data;
};

export const postKakaoLogin = async ({ code, fcmToken }: KakaoLoginReqType) => {
  const response = await api.post<KakaoLoginResType>(API_PATH.AUTH.SOCIAL, {
    code,
    fcmToken,
  });
  console.log(response);
  return response.data;
};

export const postLogout = async () => {
  const response = await api.post<LogoutResType>(API_PATH.AUTH.LOGOUT);
  return response.data;
};
