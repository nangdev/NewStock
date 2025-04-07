import * as Notifications from 'expo-notifications';
import { useRouter } from 'expo-router';
import { useEffect, useRef } from 'react';
import { Platform } from 'react-native';
import Toast from 'react-native-toast-message';
import { registerForPushNotifications } from 'utils/pushNotification';

export default function useFCMNotifications() {
  const router = useRouter();
  const notificationListener = useRef<Notifications.EventSubscription>();
  const responseListener = useRef<Notifications.EventSubscription>();

  useEffect(() => {
    async function setupPushNotifications() {
      await registerForPushNotifications();

      Notifications.setNotificationHandler({
        handleNotification: async () => ({
          shouldShowAlert: true,
          shouldPlaySound: true,
          shouldSetBadge: false,
        }),
      });

      // Memo: 앱이 실행 중일 때 (foreground) 알림 처리
      notificationListener.current = Notifications.addNotificationReceivedListener(
        (notification) => {
          console.log('푸시 알림 도착:', notification);
          handleNotificationAction(notification.request.content.data);
        }
      );

      // Memo: 알림을 클릭했을 때 (background, quit state)
      responseListener.current = Notifications.addNotificationResponseReceivedListener(
        (response) => {
          console.log('푸시 알림 클릭:', response);
          handleNotificationAction(response.notification.request.content.data);
        }
      );
    }

    setupPushNotifications();

    return () => {
      if (notificationListener.current)
        Notifications.removeNotificationSubscription(notificationListener.current);
      if (responseListener.current)
        Notifications.removeNotificationSubscription(responseListener.current);
    };
  }, []);

  // Memo: 앱이 종료되었다가 실행된 경우 (quit state)
  useEffect(() => {
    if (Platform.OS === 'android') {
      Notifications.getLastNotificationResponseAsync().then((response) => {
        if (response) {
          console.log('앱이 종료되었다가 실행된 경우:', response);
          handleNotificationAction(response.notification.request.content.data);
        }
      });
    }
  }, []);

  // Memo: 알림 데이터에 따라 라우팅 처리
  // Todo: 백엔드와 소통 후 데이터 타입, content 변경
  const handleNotificationAction = (data: { type?: string; newsId?: number }) => {
    if (!data?.type) return;

    switch (data.type) {
      case 'NEWS':
        if (data.newsId) {
          router.navigate(`/news/${data.newsId}`);
        }
        break;
      case 'NEWS_LETTER':
        router.navigate('/newsletter');
        break;
      default:
        Toast.show({
          type: 'error',
          text1: '알 수 없는 알림 유형입니다',
          text2: `데이터 타입: ${data.type}`,
        });

        console.warn('알 수 없는 알림 유형:', data.type);
    }
  };
}
