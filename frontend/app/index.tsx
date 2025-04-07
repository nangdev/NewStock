import CustomButton from 'components/CustomButton';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View } from 'react-native';

export default function Home() {
  const router = useRouter();

  const onPressIntro = () => {
    router.navigate(ROUTE.INTRO.INTRO);
  };

  const onPressMain = () => {
    router.navigate('/main');
  };

  const onPressMyPage = () => {
    router.navigate(ROUTE.MYPAGE);
  };

  return (
    <>
      <CustomHeader />
      <View className="flex-1 items-center justify-center gap-2">
        <CustomButton variant="semiRounded" onPress={onPressIntro}>
          소개 페이지
        </CustomButton>

        <CustomButton variant="semiRounded" onPress={onPressMain}>
          메인 페이지
        </CustomButton>

        <CustomButton variant="semiRounded" onPress={onPressMyPage}>
          마이 페이지
        </CustomButton>
      </View>
      <CustomFooter />
    </>
  );
}
