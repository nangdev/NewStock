import * as Device from 'expo-device';
import * as Notifications from 'expo-notifications';

export async function registerForPushNotifications() {
  if (!Device.isDevice) {
    console.warn('푸시 알림은 실제 기기에서만 동작합니다.');
    return 'FCM_TOKEN_INVALID';
  }

  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;
  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }
  if (finalStatus !== 'granted') {
    console.warn('푸시 알림 권한이 거부되었습니다.');
    return 'FCM_TOKEN_INVALID';
  }

  try {
    const token = (await Notifications.getExpoPushTokenAsync()).data;
    console.log('푸시 토큰:', token);
    return token;
  } catch (error) {
    console.error('푸시 토큰 가져오기 실패:', error);
    return 'FCM_TOKEN_INVALID';
  }
}
