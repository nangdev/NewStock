import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Slot } from 'expo-router';
import { ImageBackground } from 'react-native';

import '../global.css';

const queryClient = new QueryClient();

export default function Layout() {
  return (
    <ImageBackground
      source={require('../assets/background.png')}
      style={{ flex: 1 }}
      imageStyle={{ resizeMode: 'cover' }}>
      <QueryClientProvider client={queryClient}>
        <Slot />
      </QueryClientProvider>
    </ImageBackground>
  );
}
