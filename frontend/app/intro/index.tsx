import { AntDesign } from '@expo/vector-icons';
import CustomButton from 'components/CustomButton';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View, Image } from 'react-native';

export default function Intro() {
  const router = useRouter();

  const onPressLogin = () => {
    router.navigate(ROUTE.USER.LOGIN);
  };

  return (
    <View className="flex-1 items-center justify-center">
      <Image
        source={require('../../assets/splash.png')}
        style={{ width: 250, height: 250, resizeMode: 'contain' }}
      />

      <CustomButton variant="semiRounded" onPress={onPressLogin} className="h-[45px] gap-4 px-10">
        맞춤 주식 뉴스 보러 가기
        <AntDesign name="arrowright" size={14} />
      </CustomButton>
    </View>
  );
}
