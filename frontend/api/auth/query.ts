import { useMutation } from '@tanstack/react-query';
import { useUserInfoMutation } from 'api/user/query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { removeToken, setToken } from 'utils/token';

import { postKakaoLogin, postLogin, postLogout } from '.';

export const useLoginMutation = () => {
  const { mutate } = useUserInfoMutation();

  return useMutation({
    mutationFn: postLogin,
    onSuccess: async (data) => {
      const { accessToken, refreshToken } = data.data;

      await setToken({ key: 'accessToken', value: accessToken });
      await setToken({ key: 'refreshToken', value: refreshToken });

      mutate();
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};

export const useKakaoLoginMutation = () => {
  const { mutate } = useUserInfoMutation();

  return useMutation({
    mutationFn: postKakaoLogin,
    onSuccess: async (data) => {
      const { accessToken, refreshToken } = data.data;

      await setToken({ key: 'accessToken', value: accessToken });
      await setToken({ key: 'refreshToken', value: refreshToken });

      mutate();
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};

export const useLogoutMutation = () => {
  const router = useRouter();

  return useMutation({
    mutationFn: postLogout,
    onSuccess: async () => {
      await removeToken('accessToken');
      await removeToken('refreshToken');

      router.navigate(ROUTE.INTRO.INTRO);
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};
