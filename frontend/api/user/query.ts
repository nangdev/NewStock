import { useMutation } from '@tanstack/react-query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import useUserStore from 'store/user';

import { getCheckEmail, getUserInfo, postSignIn, putUserRole } from '.';

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

export const useUserInfoMutation = () => {
  const router = useRouter();
  const userStore = useUserStore();

  return useMutation({
    mutationKey: ['userInfo'],
    mutationFn: getUserInfo,
    onSuccess: async (data) => {
      console.log('유저 정보 조회 성공');

      userStore.setUserInfo(data.data);

      if (!data.data.role) {
        router.navigate(ROUTE.INTRO.ONBOARDING);
      } else {
        router.navigate(ROUTE.HOME);
      }
    },
    onError: (error) => {
      console.error(error);
      router.navigate(ROUTE.INTRO.INTRO);
    },
    retry: false,
  });
};

export const useUserRoleMutation = () => {
  return useMutation({
    mutationFn: putUserRole,
    onSuccess: () => {
      console.log('유저 권한 변경 성공');
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};
