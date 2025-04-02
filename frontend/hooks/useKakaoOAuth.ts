import { useKakaoLoginMutation } from 'api/auth/query';
import * as Linking from 'expo-linking';
import { useEffect } from 'react';

export default function useKakaoOAuth() {
  const { mutate } = useKakaoLoginMutation();

  useEffect(() => {
    const handleDeepLink = (event: { url: string }) => {
      const url = event.url;
      const code = new URL(url).searchParams.get('code');

      if (code) {
        // Todo: fcmToken 연동
        mutate({
          code,
          fcmToken: 'fcmToken123',
        });
      }
    };

    const subscription = Linking.addEventListener('url', handleDeepLink);

    Linking.getInitialURL().then((url) => {
      if (url) handleDeepLink({ url });
    });

    return () => {
      subscription.remove();
    };
  }, []);
}
