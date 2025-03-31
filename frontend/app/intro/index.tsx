import { AntDesign } from '@expo/vector-icons';
import CustomButton from 'components/CustomButton';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View, Image, Text } from 'react-native';

export default function Intro() {
  const router = useRouter();

  const onPressLogin = () => {
    router.navigate(ROUTE.USER.LOGIN);
  };

  return (
    <View className="flex-1 items-center justify-center gap-12">
      <View className="items-center gap-4">
        <Image
          source={require('../../assets/logo.png')}
          style={{ width: 150, height: 150, resizeMode: 'contain' }}
        />
        <View className="items-center">
          <Text className="text-5xl font-bold text-primary">NewStock</Text>
          <Text className="text-sm">주식 알림과 뉴스 레터를 동시에!</Text>
        </View>
      </View>

      <CustomButton variant="semiRounded" onPress={onPressLogin} className="h-[45px] gap-4 px-10">
        맞춤 주식 뉴스 보러 가기
        <AntDesign name="arrowright" size={14} />
      </CustomButton>
    </View>
  );
}
