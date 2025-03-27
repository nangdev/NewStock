import { useMutation } from '@tanstack/react-query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { setToken } from 'utils/token';

import { postLogin } from '.';

export const useLoginMutation = () => {
  const router = useRouter();

  return useMutation({
    mutationFn: postLogin,
    onSuccess: async (data) => {
      const { accessToken, refreshToken } = data.data;

      await setToken({ key: 'accessToken', value: accessToken });
      await setToken({ key: 'refreshToken', value: refreshToken });

      router.navigate(ROUTE.HOME);
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};
