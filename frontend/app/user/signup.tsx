import { AntDesign } from '@expo/vector-icons';
import { useCheckEmailMutation, useSignInMutation } from 'api/user/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomButton from 'components/CustomButton';
import InputField from 'components/user/InputField';
import { ROUTE } from 'constants/routes';
import { Link } from 'expo-router';
import { useEffect, useState } from 'react';
import { Text, View } from 'react-native';

// Todo: 유효성 검사
export default function SignUp() {
  const [email, setEmail] = useState('');
  const [isChecked, setIsChecked] = useState(false);
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [nickname, setNickname] = useState('');
  const [userName, setUserName] = useState('');

  const { mutate: signInMutate } = useSignInMutation();
  const { mutate: checkEmailMutate, isSuccess } = useCheckEmailMutation();

  useEffect(() => {
    if (isSuccess) {
      setIsChecked(true);
    }
  }, [isSuccess]);

  useEffect(() => {
    setIsChecked(false);
  }, [email]);

  const onCheckEmail = () => {
    checkEmailMutate({ email });
  };

  const onSubmitSignUp = () => {
    signInMutate({
      email,
      password,
      nickname,
      userName,
    });
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

      <BlurOverlay className="items-center gap-4">
        <View className="w-full flex-row items-center gap-4">
          <InputField
            value={email}
            onChangeText={setEmail}
            placeholder="이메일"
            keyboardType="email-address"
            className={isChecked ? 'flex-[1.2]' : 'flex-1'}
          />

          {!isChecked ? (
            <View>
              <CustomButton
                variant="semiRounded"
                onPress={onCheckEmail}
                className="h-[40px] shadow-lg shadow-black">
                중복체크
              </CustomButton>
            </View>
          ) : (
            <View>
              <AntDesign name="check" size={24} color="green" />
            </View>
          )}
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
        <InputField value={userName} onChangeText={setUserName} placeholder="이름" />

        <CustomButton
          variant="semiRounded"
          onPress={onSubmitSignUp}
          disabled={!isChecked}
          className="mt-4 h-[45px] w-full flex-row items-center justify-center rounded-lg bg-primary shadow-lg shadow-black disabled:opacity-70">
          회원가입
        </CustomButton>

        <Link className="self-end text-sm text-text_gray underline" href={ROUTE.USER.LOGIN}>
          이미 회원이신가요? 로그인하기
        </Link>
      </BlurOverlay>
    </View>
  );
}
