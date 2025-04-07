import { FontAwesome, MaterialIcons, AntDesign } from '@expo/vector-icons';
import { useLogoutMutation } from 'api/auth/query';
import { useUserDeleteMutation } from 'api/user/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View, Text, TouchableOpacity } from 'react-native';
import useUserStore from 'store/user';

export default function MyPage() {
  const router = useRouter();
  const userStore = useUserStore();

  const { mutate: logoutMutate } = useLogoutMutation();
  const { mutate: userDeleteMutate } = useUserDeleteMutation();

  const menuItems = [
    {
      label: '관심 종목 수정',
      onPressItem: () => {
        router.navigate(ROUTE.SET_INTEREST);
      },
      icon: <FontAwesome name="star" size={24} color="#724EDB" />,
    },
    {
      label: '로그아웃',
      onPressItem: () => {
        logoutMutate();
      },
      icon: <MaterialIcons name="logout" size={24} color="#724EDB" />,
    },
    {
      label: '회원탈퇴',
      onPressItem: () => {
        userDeleteMutate();
      },
      icon: <AntDesign name="deleteuser" size={24} color="#724EDB" />,
    },
  ];

  return (
    <>
      <CustomHeader />
      <View className="flex-1 items-center justify-center">
        <BlurOverlay className="w-[90%] items-center gap-6 px-6 py-10">
          <View className="items-center gap-2">
            <Text className="text-md text-text_gray">
              안녕하세요
              <Text className="font-bold text-primary"> {userStore.userInfo?.nickname}</Text>님
            </Text>
          </View>

          <View className="mt-6 w-full gap-4">
            {menuItems.map((item) => (
              <TouchableOpacity
                key={item.label}
                onPress={item.onPressItem}
                className="w-full flex-row items-center justify-between rounded-lg border border-stroke bg-white px-6 py-6 shadow-md">
                <View className="flex-row items-center gap-4">
                  {item.icon}
                  <Text className="text-base text-text">{item.label}</Text>
                </View>
                <AntDesign name="right" size={16} color="#888" />
              </TouchableOpacity>
            ))}
          </View>
        </BlurOverlay>
      </View>
    </>
  );
}
