import axios from 'axios';
import { API_BASE_URL, API_PATH } from 'constants/api';
import { ROUTE } from 'constants/routes';
import { router } from 'expo-router';
import { getToken, removeToken, setToken } from 'utils/token';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
  adapter: 'fetch',
});

api.interceptors.request.use(
  async (config) => {
    const token = await getToken('accessToken');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => {
    console.log(response.data);

    return response;
  },
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.data.data === 5003 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = await getToken('refreshToken');
        const response = await axios.post(`${API_BASE_URL}/${API_PATH.AUTH.REFRESH}`, {
          refreshToken,
        });
        const newAccessToken = response.data.accessToken;
        const newRefreshToken = response.data.accessToken;

        await setToken({ key: 'accessToken', value: newAccessToken });
        await setToken({ key: 'refreshToken', value: newRefreshToken });

        originalRequest.headers.Authorization = `Bearer ${response.data.accessToken}`;

        return api(originalRequest);
      } catch (refreshError) {
        await removeToken('accessToken');
        await removeToken('refreshToken');

        // Memo: 로그인 화면으로 리다이렉트
        router.replace(ROUTE.USER.LOGIN);
        return Promise.reject(refreshError);
      }
    }

    // Memo: 개발 환경에서만 에러 로그 표시
    if (__DEV__) {
      console.error('API Error:', error.response?.data || error.message);
    }
    return Promise.reject(error);
  }
);
