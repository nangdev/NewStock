import { Ionicons } from '@expo/vector-icons';
import { ROUTE } from 'constants/routes';
import { useRouter, useNavigation } from 'expo-router';
import { View, TouchableOpacity } from 'react-native';

export default function CustomHeader() {
  const navigation = useNavigation();
  const router = useRouter();

  return (
    <View className="absolute w-full flex-row items-center justify-between px-4 py-2">
      <TouchableOpacity onPress={() => navigation.goBack()}>
        <Ionicons name="chevron-back" size={24} color="black" />
      </TouchableOpacity>

      <View className="flex-row items-center gap-2">
        <TouchableOpacity className="p-2" onPress={() => router.push('/notifications')}>
          <Ionicons name="notifications-outline" size={28} color="black" />
        </TouchableOpacity>

        <TouchableOpacity className="p-2" onPress={() => router.push(ROUTE.MYPAGE)}>
          <Ionicons name="person-circle-outline" size={32} color="black" />
        </TouchableOpacity>
      </View>
    </View>
  );
}
