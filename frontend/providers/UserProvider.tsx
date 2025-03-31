import { useUserInfoQuery } from 'api/user/query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { useEffect } from 'react';
import useUserStore from 'store/user';

export default function UserProvider({ children }: { children: React.ReactNode }) {
  const { data, isSuccess, isLoading, isError } = useUserInfoQuery();
  const userStore = useUserStore();
  const router = useRouter();

  useEffect(() => {
    SplashScreen.preventAutoHideAsync();
  }, []);

  useEffect(() => {
    if (isSuccess && data) {
      console.log('User Info:', data);
      userStore.setUserInfo(data.data);

      if (!data.data.role) {
        router.replace(ROUTE.INTRO.ONBOARDING);
      }
    }

    if (isError) {
      router.replace(ROUTE.INTRO.INTRO);
    }

    if (!isLoading) {
      SplashScreen.hideAsync();
    }
  }, [isSuccess, data, isLoading, isError, router]);

  if (isLoading) {
    return null;
  }

  return <>{children}</>;
}
