import * as SecureStore from 'expo-secure-store';
import { TokenType } from 'types/token';

export const setToken = async ({ key, value }: { key: TokenType; value: string }) => {
  await SecureStore.setItemAsync(key, value);
};

export const getToken = async (key: TokenType) => {
  const token = await SecureStore.getItemAsync(key);

  return token;
};

export const removeToken = async (key: TokenType) => {
  await SecureStore.deleteItemAsync(key);
};
