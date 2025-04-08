import { KAKAO_REDIRECT_URI } from 'constants/api';
import { useState } from 'react';
import { View, ActivityIndicator } from 'react-native';
import WebView from 'react-native-webview';

export default function KakaoLogin() {
  const [isLoading, setIsLoading] = useState(true);

  const KAKAO_AUTH_URL = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.EXPO_PUBLIC_KAKAO_CLIENT_ID}&redirect_uri=${KAKAO_REDIRECT_URI}`;

  return (
    <>
      {isLoading ? (
        <View className="h-full w-full items-center justify-center">
          <ActivityIndicator size="large" color="#724EDB" />
        </View>
      ) : (
        <View className="flex-1">
          <WebView source={{ uri: KAKAO_AUTH_URL }} onLoad={() => setIsLoading(false)} />
        </View>
      )}
    </>
  );
}
