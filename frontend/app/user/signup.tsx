import BlurOverlay from 'components/BlurOverlay';
import CustomButton from 'components/CustomButton';
import InputField from 'components/user/InputField';
import { Link } from 'expo-router';
import { useState } from 'react';
import { Text, View } from 'react-native';

export default function SignUp() {
  const [email, setEmail] = useState('');
  const [emailChecked, setEmailChecked] = useState(false);
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [nickname, setNickname] = useState('');
  const [name, setName] = useState('');

  const onCheckEmail = () => {
    // Todo: API 연결
    console.log('이메일 중복 체크:', email);
  };

  const onSubmitSignUp = () => {
    // Todo: API 연결
    console.log('회원가입');
  };

  return (
    <View className="flex-1 items-center justify-center gap-12 p-4">
      <View className="items-center gap-1">
        <Text className="text-4xl font-bold text-primary">환영합니다</Text>
        <View className="items-center">
          <Text className="text-xl font-bold text-primary">주식알림과 뉴스레터를 동시에!</Text>
          <Text className="text-xl font-bold text-primary">NewStock과 함께하세요</Text>
        </View>
      </View>

      <BlurOverlay className="gap-4">
        <View className="w-full flex-row gap-4">
          <InputField
            value={email}
            onChangeText={setEmail}
            placeholder="이메일"
            keyboardType="email-address"
            className="flex-1"
          />
          <CustomButton
            variant="semiRounded"
            onPress={onCheckEmail}
            // Todo: API 연결시 활성화
            // disabled={emailChecked}
            className="shadow-lg shadow-black">
            중복체크
          </CustomButton>
        </View>

        <InputField
          value={password}
          onChangeText={setPassword}
          placeholder="비밀번호"
          secureTextEntry
        />

        <InputField
          value={passwordConfirm}
          onChangeText={setPasswordConfirm}
          placeholder="비밀번호 확인"
          secureTextEntry
        />

        <InputField value={nickname} onChangeText={setNickname} placeholder="닉네임" />
        <InputField value={name} onChangeText={setName} placeholder="이름" />

        <CustomButton
          variant="semiRounded"
          className="mt-4 shadow-lg shadow-black"
          onPress={onSubmitSignUp}>
          회원가입
        </CustomButton>

        <Link className="text-right text-sm text-text_gray underline" href="/user/signin">
          이미 회원이신가요? 로그인하기
        </Link>
      </BlurOverlay>
    </View>
  );
}
