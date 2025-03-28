import { useMutation } from '@tanstack/react-query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';

import { getCheckEmail, postSignIn } from '.';

export const useSignInMutation = () => {
  const router = useRouter();

  return useMutation({
    mutationFn: postSignIn,
    onSuccess: (data) => {
      console.log('회원가입 성공');
      console.log(data.message);
      router.navigate(ROUTE.USER.LOGIN);
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};

export const useCheckEmailMutation = () => {
  return useMutation({
    mutationFn: getCheckEmail,
    onSuccess: () => {
      console.log('이메일 중복 체크 성공');
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};
