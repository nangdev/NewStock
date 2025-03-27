// Memo: 개발 환경에서 백엔드 서버와 통신 확인 시 대신 사용
// export const API_URL = process.env.EXPO_PUBLIC_API_URL;
export const API_URL = __DEV__
  ? process.env.EXPO_PUBLIC_DEV_API_URL
  : process.env.EXPO_PUBLIC_API_URL;
export const API_VERSION = process.env.EXPO_PUBLIC_API_VERSION;
export const API_BASE_URL = `${API_URL}/${API_VERSION}`;

const DOMAIN = {
  USER: 'users',
  AUTH: 'auth',
};

export const API_PATH = {
  USER: {
    INFO: DOMAIN.USER,
    SIGN_IN: DOMAIN.USER,
    WITHDRAW: DOMAIN.USER,
    NEW: `${DOMAIN.USER}/new`,
  },
  AUTH: {
    SOCIAL: `${DOMAIN.AUTH}/social-login`,
    CHECK_EMAIL: `${DOMAIN.AUTH}/social-login/check-email`,
    LOGIN: `${DOMAIN.AUTH}/social-login/login`,
    LOGOUT: `${DOMAIN.AUTH}/social-login/logout`,
    ACCESS: `${DOMAIN.AUTH}/social-login/access`,
    REFRESH: `${DOMAIN.AUTH}/social-login/refresh`,
  },
};
