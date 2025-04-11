import { AntDesign } from '@expo/vector-icons';
import { useUserRoleMutation } from 'api/user/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomButton from 'components/CustomButton';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View, Text, Image } from 'react-native';
import useUserStore from 'store/user';

type InitProps = {
  onPressNextStep: () => void;
};

export default function Init({ onPressNextStep }: InitProps) {
  const router = useRouter();
  const userStore = useUserStore();
  const { nickname } = userStore.userInfo!;
  const { mutate } = useUserRoleMutation();

  const onPressExit = () => {
    mutate();
    router.navigate(ROUTE.HOME);
  };

  return (
    <BlurOverlay className="w-full items-center gap-8 p-8 py-24">
      <View className="items-center gap-2">
        <Text className="text-xl font-bold text-text_gray">
          <Text className="text-xl font-bold text-primary">{nickname}</Text>님, 처음 오셨군요!
        </Text>
        <Text className="text-xl font-bold text-text_gray">관심 종목을 설정해주시면</Text>
        <Text className="text-xl font-bold text-text_gray">맞춤 알림을 보내드려요!</Text>
      </View>

      <View className="w-full rounded-lg border border-stroke bg-white p-4 shadow-lg shadow-black">
        <View className="flex-row items-center border-b border-stroke pb-4">
          <Image
            source={require('../../../assets/image/sample_samsung.png')}
            className="mr-3 h-12 w-12 rounded-lg"
          />
          <View className="flex-1 flex-row items-center">
            <View className="flex-1 justify-center gap-1">
              <Text className="font-bold">삼성전자</Text>
              <Text className="text-xs text-text_gray">005930</Text>
            </View>
            <View className="items-end gap-1">
              <Text>77,777원</Text>
              <Text className="text-xs text-red-500">+7.77%</Text>
            </View>
          </View>
        </View>

        <View>
          <View className="flex-row items-center justify-between border-b border-stroke py-4">
            <View className="flex-row items-center gap-3">
              <AntDesign name="smileo" size={24} color="red" />
              <Text className="text-sm">삼성전자에 대한 호재 뉴스 제목 예시</Text>
            </View>
            <Text className="text-xs text-text_gray">1시간 전</Text>
          </View>
          <View className="flex-row items-center justify-between pt-4">
            <View className="flex-row items-center gap-3">
              <AntDesign name="frowno" size={24} color="blue" />
              <Text className="text-sm">삼성전자에 대한 악재 뉴스 제목 예시</Text>
            </View>
            <Text className="text-xs text-text_gray">1시간 전</Text>
          </View>
        </View>
      </View>

      <View className="w-full items-center gap-2">
        <CustomButton variant="semiRounded" className="h-[45px] w-[70%]" onPress={onPressNextStep}>
          관심 종목 설정하러 가기
        </CustomButton>
        <CustomButton
          variant="ghost"
          className="h-[45px] w-[70%] rounded-lg border border-primary"
          onPress={onPressExit}>
          서비스 이용하러 가기
        </CustomButton>
      </View>
    </BlurOverlay>
  );
}
