import { UserInfoType } from 'types/user';
import { create } from 'zustand';

type UserStoreType = {
  userInfo: UserInfoType | null;
  setUserInfo: (userInfo: UserInfoType | null) => void;
  getUserInfo: () => void;
};

const useUserStore = create<UserStoreType>((set, get) => ({
  userInfo: null,
  setUserInfo: (userInfo) => set({ userInfo }),
  getUserInfo: () => get().userInfo,
}));

export default useUserStore;
