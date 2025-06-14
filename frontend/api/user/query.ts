import { useMutation } from '@tanstack/react-query';
import { useLogoutMutation } from 'api/auth/query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import Toast from 'react-native-toast-message';
import useUserStore from 'store/user';

import {
  deleteUser,
  getCheckEmail,
  getUserInfo,
  postSignUp,
  postVerifyCheck,
  postVerifySend,
  putUserNickname,
  putUserRole,
} from '.';

export const useSignUpMutation = () => {
  const router = useRouter();

  return useMutation({
    mutationFn: postSignUp,
    onSuccess: () => {
      Toast.show({ type: 'success', text1: '회원가입에 성공했습니다' });

      router.navigate(ROUTE.USER.LOGIN);
    },
    onError: (error) => {
      Toast.show({ type: 'error', text1: '회원가입 중 오류가 발생했습니다' });
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
      console.error(error);
      Toast.show({
        type: 'error',
        text1: '유저 권한 변경 중 오류가 발생했습니다',
      });
    },
  });
};

export const useUserNicknameMutation = () => {
  const { setUserInfo, userInfo } = useUserStore();

  return useMutation({
    mutationFn: putUserNickname,
    onSuccess: (data) => {
      console.log('유저 닉네임 변경 성공');
      setUserInfo({
        userId: userInfo!.userId,
        email: userInfo!.email,
        role: userInfo!.role,
        nickname: data.data.nickname,
      });
      Toast.show({
        type: 'success',
        text1: '닉네임이 성공적으로 변경되었습니다!',
      });
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
      Toast.show({
        type: 'error',
        text1: '닉네임 변경 중 오류가 발생했습니다',
      });
    },
  });
};

export const useUserDeleteMutation = () => {
  const { mutate } = useLogoutMutation();

  return useMutation({
    mutationFn: deleteUser,
    onSuccess: () => {
      console.log('회원 탈퇴 성공');
      mutate();
    },
    onError: (error) => {
      console.error(error);
      Toast.show({
        type: 'error',
        text1: '회원 탈퇴 중 오류가 발생했습니다',
      });
    },
  });
};

export const useVerifySendMutation = () => {
  return useMutation({
    mutationFn: postVerifySend,
    onSuccess: () => {
      Toast.show({
        type: 'success',
        text1: '인증 번호가 전송되었습니다',
      });
    },
    onError: () => {
      Toast.show({
        type: 'error',
        text1: '인증 번호 전송 중 오류가 발생했습니다',
      });
    },
  });
};

export const useVerifyCheckMutation = () => {
  return useMutation({
    mutationFn: postVerifyCheck,
    onError: (error) => {
      // Todo: 에러 처리
      console.log(error);
      Toast.show({
        type: 'error',
        text1: error.message,
      });
    },
  });
};
