import { FontAwesome, MaterialIcons, AntDesign, Feather } from '@expo/vector-icons';
import { useLogoutMutation } from 'api/auth/query';
import { useUserDeleteMutation, useUserNicknameMutation } from 'api/user/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomButton from 'components/CustomButton';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { View, Text, TouchableOpacity, Modal, TextInput } from 'react-native';
import useUserStore from 'store/user';

export default function MyPage() {
  const router = useRouter();
  const userStore = useUserStore();

  const { mutate: logoutMutate } = useLogoutMutation();
  const { mutate: userDeleteMutate } = useUserDeleteMutation();
  const { mutate: changeNickname } = useUserNicknameMutation();

  const [isModalVisible, setModalVisible] = useState(false);
  const [nicknameInput, setNicknameInput] = useState(userStore.userInfo?.nickname ?? '');

  const handleConfirmNicknameChange = () => {
    if (nicknameInput.trim()) {
      changeNickname({ nickname: nicknameInput.trim() });
      setModalVisible(false);
    }
  };

  const menuItems = [
    {
      label: '닉네임 변경',
      onPressItem: () => {
        setNicknameInput(userStore.userInfo?.nickname ?? '');
        setModalVisible(true);
      },
      icon: <Feather name="edit-2" size={24} color="#724EDB" />,
    },
    {
      label: '관심 종목 수정',
      onPressItem: () => {
        router.navigate(ROUTE.SET_INTEREST);
      },
      icon: <FontAwesome name="star" size={24} color="#724EDB" />,
    },
    {
      label: '로그아웃',
      onPressItem: () => {
        logoutMutate();
      },
      icon: <MaterialIcons name="logout" size={24} color="#724EDB" />,
    },
    {
      label: '회원탈퇴',
      onPressItem: () => {
        userDeleteMutate();
      },
      icon: <AntDesign name="deleteuser" size={24} color="#724EDB" />,
    },
  ];

  return (
    <>
      <CustomHeader title="마이 페이지" />
      <View className="flex-1 items-center justify-center">
        <BlurOverlay className="w-[90%] items-center gap-6 px-6 py-10">
          <View className="items-center gap-2">
            <Text className="text-md text-text_gray">
              안녕하세요
              <Text className="font-bold text-primary"> {userStore.userInfo?.nickname}</Text>님
            </Text>
          </View>

          <View className="mt-6 w-full gap-4">
            {menuItems.map((item) => (
              <TouchableOpacity
                key={item.label}
                onPress={item.onPressItem}
                className="w-full flex-row items-center justify-between rounded-lg border border-stroke bg-white px-6 py-6 shadow-md">
                <View className="flex-row items-center gap-4">
                  {item.icon}
                  <Text className="text-base text-text">{item.label}</Text>
                </View>
                <AntDesign name="right" size={16} color="#888" />
              </TouchableOpacity>
            ))}
          </View>
        </BlurOverlay>
      </View>
      <Modal
        visible={isModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setModalVisible(false)}>
        <View className="flex-1 items-center justify-center bg-black/40">
          <View className="w-[90%] rounded-2xl bg-white px-6 py-8 shadow-lg">
            <Text className="mb-4 text-lg font-bold text-black">닉네임 변경</Text>

            <TextInput
              value={nicknameInput}
              onChangeText={setNicknameInput}
              placeholder="닉네임은 2~10자여야 합니다"
              maxLength={10}
              className="rounded-lg border border-stroke px-4 py-3 text-base"
            />

            <View className="mt-6 flex-row justify-end gap-2">
              <CustomButton
                variant="semiRounded"
                onPress={() => setModalVisible(false)}
                className="h-[40px] border border-gray-300 bg-white px-6">
                <Text className="text-sm text-gray-600">취소</Text>
              </CustomButton>

              <CustomButton
                variant="semiRounded"
                disabled={nicknameInput.trim().length < 2 || nicknameInput.trim().length > 10}
                onPress={handleConfirmNicknameChange}
                className="h-[40px] bg-primary px-6 disabled:opacity-50">
                <Text className="text-sm font-bold text-white">확인</Text>
              </CustomButton>
            </View>
          </View>
        </View>
      </Modal>
      <CustomFooter />
    </>
  );
}
