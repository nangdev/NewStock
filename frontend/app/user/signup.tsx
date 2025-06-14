import { AntDesign } from '@expo/vector-icons';
import { useSignUpMutation, useVerifyCheckMutation, useVerifySendMutation } from 'api/user/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomButton from 'components/CustomButton';
import InputField from 'components/user/InputField';
import { REGEX } from 'constants/regex';
import { ROUTE } from 'constants/routes';
import { Link } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Text, View } from 'react-native';
import Toast from 'react-native-toast-message';

export default function SignUp() {
  const [email, setEmail] = useState('');
  const [isChecked, setIsChecked] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [nickname, setNickname] = useState('');
  const [authCode, setAuthCode] = useState('');
  const [errorField, setErrorField] = useState<string | null>(null);

  const { mutate: signUpMutate } = useSignUpMutation();
  const { mutate: sendVerifyMutate, isPending } = useVerifySendMutation();
  const { mutate: checkVerifyMutate } = useVerifyCheckMutation();

  useEffect(() => {
    setIsChecked(false);
  }, [email]);

  const validateInputs = (): boolean => {
    if (!REGEX.SIGN_UP.EMAIL.test(email)) {
      setErrorField('email');
      Toast.show({ type: 'error', text1: '올바른 이메일을 입력해주세요' });
      return false;
    }
    if (!REGEX.SIGN_UP.PASSWORD.test(password)) {
      setErrorField('password');
      Toast.show({
        type: 'error',
        text1: '비밀번호는 숫자와 특수문자 포함 8~20자여야 합니다',
      });
      return false;
    }
    if (password !== passwordConfirm) {
      setErrorField('passwordConfirm');
      Toast.show({ type: 'error', text1: '비밀번호가 일치하지 않습니다' });
      return false;
    }
    if (!REGEX.SIGN_UP.NAME.test(nickname)) {
      setErrorField('nickname');
      Toast.show({ type: 'error', text1: '닉네임은 2~10자여야 합니다' });
      return false;
    }
    return true;
  };

  const onSendVerifyEmail = () => {
    if (!REGEX.SIGN_UP.EMAIL.test(email)) {
      setErrorField('email');
      Toast.show({ type: 'error', text1: '올바른 이메일을 입력해주세요' });
      return;
    }
    sendVerifyMutate(
      {
        email,
      },
      {
        onSuccess: () => {
          setIsChecking(true);
        },
      }
    );
  };

  const onCheckVerifyEmail = () => {
    checkVerifyMutate(
      {
        email,
        authCode,
      },
      {
        onSuccess: () => {
          setIsChecked(true);
          setIsChecking(false);
        },
        onError: () => {
          setIsChecked(false);
          setIsChecking(false);
        },
      }
    );
  };

  const onSubmitSignUp = (): void => {
    if (!validateInputs()) return;
    signUpMutate({ email, password, nickname });
  };

  return (
    <View className="flex-1 items-center justify-center gap-6 p-4">
      <View className="items-center gap-1">
        <Text className="text-4xl font-bold text-primary">환영합니다</Text>
        <Text className="text-xl font-bold text-primary">주식알림과 뉴스레터를 동시에!</Text>
        <Text className="text-xl font-bold text-primary">NewStock과 함께하세요</Text>
      </View>

      <BlurOverlay className="items-center gap-4">
        {!isChecked ? (
          <View className="w-full flex-row items-center gap-4">
            <InputField
              value={email}
              onChangeText={setEmail}
              placeholder="이메일을 입력해주세요"
              keyboardType="email-address"
              className={`flex-1 ${errorField === 'email' ? 'border-red-500' : ''}`}
            />
            <CustomButton
              variant="semiRounded"
              onPress={isChecking ? onCheckVerifyEmail : onSendVerifyEmail}
              className="h-[40px] shadow-lg">
              {isChecking ? (
                '인증완료'
              ) : isPending ? (
                <ActivityIndicator color="white" />
              ) : (
                '인증하기'
              )}
            </CustomButton>
          </View>
        ) : (
          <View className="w-full flex-row items-center gap-4">
            <InputField
              value={email}
              onChangeText={setEmail}
              placeholder="이메일을 입력해주세요"
              keyboardType="email-address"
              className="flex-[1.2]"
            />
            <AntDesign name="check" size={24} color="green" />
          </View>
        )}

        {isChecking && (
          <InputField
            value={authCode}
            onChangeText={setAuthCode}
            placeholder="인증번호를 입력해주세요"
          />
        )}

        <InputField
          value={password}
          onChangeText={setPassword}
          placeholder="비밀번호는 숫자와 특수문자 포함 8~20자를 입력해주세요"
          secureTextEntry
          className={errorField === 'password' ? 'border-red-500' : ''}
        />

        <InputField
          value={passwordConfirm}
          onChangeText={setPasswordConfirm}
          placeholder="비밀번호를 한 번 더 입력해주세요"
          secureTextEntry
          className={errorField === 'passwordConfirm' ? 'border-red-500' : ''}
        />

        <InputField
          value={nickname}
          onChangeText={setNickname}
          placeholder="닉네임은 2~10자여야 합니다"
          className={errorField === 'nickname' ? 'border-red-500' : ''}
        />

        <CustomButton
          variant="semiRounded"
          onPress={onSubmitSignUp}
          disabled={!isChecked}
          className="mt-4 h-[45px] w-full flex-row items-center justify-center rounded-lg bg-primary shadow-lg shadow-black disabled:opacity-70">
          회원가입
        </CustomButton>

        <Link className="self-end py-2 text-sm text-text_gray underline" href={ROUTE.USER.LOGIN}>
          이미 회원이신가요? 로그인하기
        </Link>
      </BlurOverlay>
    </View>
  );
}
