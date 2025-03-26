import axios from 'axios';
import { API_URL } from 'constants/api';
import { getToken } from 'utils/token';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
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
  (response) => response,
  async (error) => {
    // const originalRequest = error.config;

    // Todo: 토큰 만료 API 작업 후 주석 해제 및 수정 (예: 401 또는 커스텀 에러 코드)
    // if (error.response?.status === 401 && !originalRequest._retry) {
    //   originalRequest._retry = true;

    //   try {
    //     const refreshToken = await getToken('refreshToken');
    //     // Todo: API 명세서 완성되면 수정
    //     const response = await axios.post(`${API_URL}/auth/refresh`, { refreshToken });
    //     const newToken = response.data.accessToken;

    //     await setToken({ key: 'accessToken', value: newToken });

    //     originalRequest.headers.Authorization = `Bearer ${response.data.accessToken}`;

    //     return api(originalRequest);
    //   } catch (refreshError) {
    //     await removeToken('accessToken');
    //     await removeToken('refreshToken');

    //     // Memo: 로그인 화면으로 리다이렉트 시킬것인지
    //     // navigation.navigate('Login');

    //     return Promise.reject(refreshError);
    //   }
    // }

    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);
