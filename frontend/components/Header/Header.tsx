import { Ionicons } from '@expo/vector-icons';
import { ROUTE } from 'constants/routes';
import { BlurView } from 'expo-blur';
import { useRouter, useNavigation } from 'expo-router';
import { View, TouchableOpacity, Text } from 'react-native';
import useUserStore from 'store/user';

import NotificationButton from './NotificationButton';

type HeaderProps = {
  title?: string;
  onGoBack?: () => void;
  disabled?: boolean;
};

export default function CustomHeader({ title, onGoBack, disabled = false }: HeaderProps) {
  const navigation = useNavigation();
  const router = useRouter();
  const userStore = useUserStore();

  const nickname = userStore.userInfo?.nickname;
  const displayName = nickname ? nickname.slice(0, 2) : null;

  const onPressMyPage = () => {
    if (userStore.userInfo?.nickname) {
      router.push(ROUTE.MYPAGE);
    } else {
      router.push(ROUTE.INTRO.INTRO);
    }
  };

  return (
    <BlurView
      intensity={50}
      tint="light"
      className="absolute top-0 z-10 w-full flex-row items-center justify-between border-b border-gray-200 px-4 py-3 backdrop-blur-md">
      <TouchableOpacity
        disabled={disabled}
        onPress={onGoBack ? onGoBack : () => navigation.goBack()}
        className="p-1 disabled:w-0">
        <Ionicons name="chevron-back" size={26} color="black" />
      </TouchableOpacity>

      <View className="absolute left-0 right-0 flex-row items-center justify-center">
        <Text className="text-xl font-bold text-text">{title}</Text>
      </View>

      <View className="flex-row items-center gap-3">
        <NotificationButton />

        <TouchableOpacity
          onPress={onPressMyPage}
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
