import { View, Image } from 'react-native';

export default function Home() {
  return (
    <View className="flex-1 items-center justify-center">
      <Image source={require('../assets/logo.png')} style={{ width: 200, height: 200 }} />
    </View>
  );
}
