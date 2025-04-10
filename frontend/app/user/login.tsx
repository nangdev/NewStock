import { useLoginMutation } from 'api/auth/query';
import BlurOverlay from 'components/BlurOverlay';
import InputField from 'components/user/InputField';
import { ROUTE } from 'constants/routes';
import { Link, useRouter } from 'expo-router';
import { useState } from 'react';
import { Text, View, Image, TouchableOpacity } from 'react-native';
import { registerForPushNotifications } from 'utils/pushNotification';

export default function Login() {
  const router = useRouter();
  const [email, setEmail] = useState('gmlgml5023@naver.com');
  const [password, setPassword] = useState('asd123123!');
  const { mutate } = useLoginMutation();

  const onPressLogo = () => {
    router.navigate(ROUTE.INTRO.INTRO);
  };

  const onPressLogin = async () => {
    const fcmToken = await registerForPushNotifications();

    mutate({ email, password, fcmToken });
  };

  const onPressKakaoLogin = () => {
    router.navigate(ROUTE.USER.OAUTH);
  };

  return (
    <View className="flex-1 items-center justify-center gap-8 p-4">
      <TouchableOpacity onPress={onPressLogo}>
        <Text className="text-5xl font-bold text-logo">NewStock</Text>
      </TouchableOpacity>

      <BlurOverlay className="items-center gap-8">
        <View className="w-full gap-4">
          <InputField
            value={email}
            onChangeText={setEmail}
            placeholder="이메일"
            keyboardType="email-address"
            className="w-full"
          />

          <InputField
            value={password}
            onChangeText={setPassword}
            placeholder="비밀번호"
            secureTextEntry
            className="w-full"
          />
        </View>

        <View className="gap-2">
          <TouchableOpacity
            className="h-[45px] w-[300px] flex-row items-center justify-center rounded-lg bg-primary"
            onPress={onPressLogin}>
            <Text className="text-base font-bold text-white">로그인</Text>
          </TouchableOpacity>

          <TouchableOpacity onPress={onPressKakaoLogin}>
            <Image source={require('../../assets/image/kakao_login.png')} />
          </TouchableOpacity>

          <Link className="self-end py-2 text-sm text-text_gray underline" href={ROUTE.USER.SIGNUP}>
            회원이 아니신가요? 회원가입
          </Link>
        </View>
      </BlurOverlay>
    </View>
  );
}
