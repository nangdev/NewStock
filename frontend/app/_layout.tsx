import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Slot } from 'expo-router';
import { makeServer } from 'mocks/mockServer';
import UserProvider from 'providers/UserProvider';
import { useEffect } from 'react';
import { ImageBackground } from 'react-native';
import '../global.css';
import Toast from 'react-native-toast-message';

const queryClient = new QueryClient();

export default function Layout() {
  // Memo: 백엔드 서버와 API 통신 테스트시 주석처리
  // useEffect(() => {
  //   let server = null;

  //   if (__DEV__) {
  //     server = makeServer();
  //   }

  //   return () => {
  //     server.shutdown();
  //   };
  // }, []);

  return (
    <ImageBackground
      source={require('../assets/background.png')}
      style={{ flex: 1 }}
      imageStyle={{ resizeMode: 'cover' }}>
      <QueryClientProvider client={queryClient}>
        <UserProvider>
          <Slot />
          <Toast />
        </UserProvider>
      </QueryClientProvider>
    </ImageBackground>
  );
}
