// Memo: 개발 환경에서 백엔드 서버와 통신 확인 시 대신 사용
export const API_URL = process.env.EXPO_PUBLIC_API_URL;
export const API_VERSION = process.env.EXPO_PUBLIC_API_VERSION;
export const API_BASE_URL = `${API_URL}/${API_VERSION}`;
export const KAKAO_REDIRECT_URI = `${process.env.EXPO_PUBLIC_API_URL}/api/kakao-redirect.html`;

const DOMAIN = {
  USER: 'users',
  AUTH: 'auth',
  STOCK: 'stock',
  NEWS: 'news',
  NOTIFICATION: 'notification',
  NEWSLETTER: 'newsletter',
};

/** Memo: API 명세서 도메인 및 endpoint 구조 적용 */
export const API_PATH = {
  USER: {
    INFO: DOMAIN.USER,
    CHECK_EMAIL: `${DOMAIN.USER}/check-email`,
    SIGN_UP: DOMAIN.USER,
    WITHDRAW: DOMAIN.USER,
    NEW: `${DOMAIN.USER}/new`,
    NICKNAME: `${DOMAIN.USER}/nickname`,
    VERIFY_SEND: `${DOMAIN.USER}/send-email`,
    VERIFY_CHECK: `${DOMAIN.USER}/verify-email`,
  },
  AUTH: {
    SOCIAL: `${DOMAIN.AUTH}/oauth/kakao/login`,
    LOGIN: `${DOMAIN.AUTH}/login`,
    LOGOUT: `${DOMAIN.AUTH}/logout`,
    ACCESS: `${DOMAIN.AUTH}/access`,
    REFRESH: `${DOMAIN.AUTH}/refresh`,
  },
  STOCK: {
    ALL: `${DOMAIN.STOCK}`,
    DETAIL: (stockId: number) => `${DOMAIN.STOCK}/info/${stockId}`,
    USER_STOCK: `${DOMAIN.STOCK}/interest`,
    USER_STOCK_EDIT: `${DOMAIN.STOCK}/interest`,
  },
  NEWS: {
    TOP: (stockId: number) => `${DOMAIN.NEWS}/top/${stockId}`,
    STOCK_NEWS: `${DOMAIN.NEWS}`,
    DETAIL: (newsId: number) => `${DOMAIN.NEWS}/${newsId}`,
    SCRAP: `${DOMAIN.NEWS}/scrap`,
    SCRAP_ADD: (newsId: number) => `${DOMAIN.NEWS}/scrap/${newsId}`,
    SCRAP_DELETE: (newsId: number) => `${DOMAIN.NEWS}/scrap/${newsId}`,
  },
  NOTIFICATION: {
    ALL: `${DOMAIN.NOTIFICATION}`,
    DELETE: `${DOMAIN.NOTIFICATION}`,
    READ: `${DOMAIN.NOTIFICATION}`,
  },
  NEWSLETTER: `${DOMAIN.NEWSLETTER}`,
};
