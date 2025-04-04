import { useKakaoLoginMutation } from 'api/auth/query';
import * as Linking from 'expo-linking';
import { useEffect } from 'react';
import { registerForPushNotifications } from 'utils/pushNotification';

export default function useKakaoOAuth() {
  const { mutate } = useKakaoLoginMutation();

  useEffect(() => {
    const handleDeepLink = async (event: { url: string }) => {
      const url = event.url;
      const code = new URL(url).searchParams.get('code');
      const fcmToken = await registerForPushNotifications();

      if (code) {
        mutate({
          code,
          fcmToken,
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
