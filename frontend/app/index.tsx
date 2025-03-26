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

  const onPressLogin = () => {
    router.navigate(ROUTE.USER.LOGIN);
  };

  const onPressIntro = () => {
    router.navigate(ROUTE.INTRO);
  };

  const onPressMain = () => {
    router.navigate('/main');
  };

  return (
    <View className="flex-1 items-center justify-center gap-2">
      <Image
        source={require('../assets/logo.png')}
        style={{ width: 200, height: 200, resizeMode: 'contain' }}
      />

      <CustomButton variant="semiRounded" onPress={onPressSignUp}>
        <AntDesign name="user" size={24} />
        회원가입
      </CustomButton>

      <CustomButton variant="semiRounded" onPress={onPressLogin}>
        로그인
      </CustomButton>

      <CustomButton variant="semiRounded" onPress={onPressIntro}>
        소개 페이지
      </CustomButton>

      <CustomButton variant="semiRounded" onPress={onPressMain}>
        메인 페이지
      </CustomButton>
    </View>
  );
}
