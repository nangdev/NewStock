import { View, Text, TouchableOpacity } from 'react-native';
import { FontAwesome, Ionicons, MaterialIcons, AntDesign } from '@expo/vector-icons'; // 아이콘 라이브러리 추가
import BlurOverlay from 'components/BlurOverlay';
import { useRouter } from 'expo-router';
import { ROUTE } from 'constants/routes';

const onPress = () => {
  console.log('임시 기능');
};

const menuItems = [
  {
    label: '관심 종목 수정',
    onPress: onPress,
    icon: <FontAwesome name="star" size={24} color="#724EDB" />,
  },

  {
    label: '로그아웃',
    onPress: onPress,
    icon: <MaterialIcons name="logout" size={24} color="#724EDB" />,
  },
  {
    label: '회원탈퇴',
    onPress: onPress,
    icon: <AntDesign name="deleteuser" size={24} color="#724EDB" />,
  },
];

export default function MyPageScreen() {
  const router = useRouter();
  const nickname = '유보형'; // 실제 사용자 닉네임으로 대체

  const handleMenuPress = (route: string) => {
    if (route === 'logout') {
      // 로그아웃 로직
    } else if (route === 'withdraw') {
      // 회원탈퇴 로직
    } else {
      router.push(route);
    }
  };

  return (
    <View className="flex-1 items-center justify-center">
      <BlurOverlay className="w-[90%] items-center gap-6 px-6 py-10">
        <View className="items-center gap-2">
          <Text className="text-xl font-bold text-primary">{nickname}님</Text>
          <Text className="text-md text-text_gray">안녕하세요 👋</Text>
        </View>

        {/* 메뉴 리스트 */}
        <View className="mt-6 w-full gap-4">
          {menuItems.map((item) => (
            <TouchableOpacity
              key={item.label}
              onPress={item.onPress}
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
  );
}
