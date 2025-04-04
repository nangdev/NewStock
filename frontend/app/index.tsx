import { AntDesign } from '@expo/vector-icons';
import { useLogoutMutation } from 'api/auth/query';
import CustomButton from 'components/CustomButton';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View, Image } from 'react-native';

export default function Home() {
  const router = useRouter();
  const { mutate } = useLogoutMutation();

  const onPressSignUp = () => {
    router.navigate(ROUTE.USER.SIGNUP);
  };

  const onPressLogin = () => {
    router.navigate(ROUTE.USER.LOGIN);
  };

  const onPressIntro = () => {
    router.navigate(ROUTE.INTRO.INTRO);
  };

  const onPressMain = () => {
    router.navigate('/main');
  };

  const onPressInterest = () => {
    router.navigate('/intro/onboarding');
  };

  const onPressLogout = () => {
    mutate();
  };

  const onPressMyPage = () => {
    router.navigate('/mypage');
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

      <CustomButton variant="semiRounded" onPress={onPressInterest}>
        관심 종목 설정
      </CustomButton>

      <CustomButton variant="semiRounded" onPress={onPressLogout}>
        로그아웃
      </CustomButton>

      <CustomButton variant="semiRounded" onPress={onPressMyPage}>
        마이 페이지
      </CustomButton>
    </View>
  );
}
