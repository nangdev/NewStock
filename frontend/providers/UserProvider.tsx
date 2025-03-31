import { useUserInfoMutation } from 'api/user/query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { useEffect } from 'react';
import { getToken } from 'utils/token';

SplashScreen.preventAutoHideAsync();

export default function UserProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { mutateAsync } = useUserInfoMutation();

  useEffect(() => {
    const prepare = async () => {
      try {
        const token = await getToken('accessToken');

        if (token) {
          await mutateAsync();
        } else {
          router.navigate(ROUTE.INTRO.INTRO);
        }
      } catch (e) {
        console.warn('사용자 정보 로딩 오류:', e);
      } finally {
        await SplashScreen.hideAsync();
      }
    };

    prepare();
  }, []);

  return <>{children}</>;
}
