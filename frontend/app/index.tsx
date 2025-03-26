import { AntDesign } from '@expo/vector-icons';
import CustomButton from 'components/CustomButton';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View, Image } from 'react-native';

export default function Home() {
  const router = useRouter();

  const onPressSignUp = () => {
    router.navigate(ROUTE.USER.SIGNUP);
  };

  const onPressMain = () => {
    router.navigate('/main');
  };

  return (
    <View className="flex-1 items-center justify-center gap-2">
      <Image source={require('../assets/logo.png')} style={{ width: 200, height: 200 }} />

      <CustomButton variant="semiRounded" onPress={onPressSignUp}>
        <AntDesign name="user" size={24} />
        회원가입
      </CustomButton>

      <CustomButton variant="semiRounded" onPress={onPressMain}>
        메인 페이지
      </CustomButton>
    </View>
  );
}
