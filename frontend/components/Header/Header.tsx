import { Ionicons } from '@expo/vector-icons';
import { ROUTE } from 'constants/routes';
import { BlurView } from 'expo-blur';
import { useRouter, useNavigation } from 'expo-router';
import { View, TouchableOpacity, Text } from 'react-native';
import useUserStore from 'store/user';

import NotificationButton from './NotificationButton';

export default function CustomHeader() {
  const navigation = useNavigation();
  const router = useRouter();
  const userStore = useUserStore();

  const nickname = userStore.userInfo?.nickname;
  const displayName = nickname ? nickname.slice(0, 2) : null;

  return (
    <BlurView
      intensity={50}
      tint="light"
      className="absolute z-10 w-full flex-row items-center justify-between border-b border-gray-200 px-4 py-3 backdrop-blur-md">
      <TouchableOpacity onPress={() => navigation.goBack()} className="p-1">
        <Ionicons name="chevron-back" size={26} color="black" />
      </TouchableOpacity>

      <View className="flex-row items-center gap-3">
        <NotificationButton />

        <TouchableOpacity
          onPress={() => router.navigate(ROUTE.MYPAGE)}
          className="h-8 w-8 items-center justify-center overflow-hidden rounded-full bg-neutral-800">
          {displayName ? (
            <Text className="text-sm font-semibold text-white">{displayName}</Text>
          ) : (
            <Ionicons name="person" size={24} color="white" />
          )}
        </TouchableOpacity>
      </View>
    </BlurView>
  );
}
