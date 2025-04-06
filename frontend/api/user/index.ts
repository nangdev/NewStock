import { API_PATH } from 'constants/api';
import {
  EmailCheckReqType,
  EmailCheckResType,
  SignUpReqType,
  SignUpResType,
  UserInfoResType,
  UserNicknameReqType,
  UserNicknameResType,
} from 'types/api/user';

import { api } from '../axiosInstance';

export const postSignUp = async ({ email, password, nickname }: SignUpReqType) => {
  const response = await api.post<SignUpResType>(API_PATH.USER.SIGN_UP, {
    email,
    password,
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

export const putUserRole = async () => {
  const response = await api.put(API_PATH.USER.NEW);
  return response.data;
};

export const putUserNickname = async ({ nickname }: UserNicknameReqType) => {
  const response = await api.put<UserNicknameResType>(API_PATH.USER.NICKNAME, { nickname });
  return response.data;
};

export const deleteUser = async () => {
  const response = await api.delete(API_PATH.USER.WITHDRAW);
  return response.data;
};
