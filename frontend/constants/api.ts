// Memo: 개발 환경에서 백엔드 서버와 통신 확인 시 대신 사용
export const API_URL = process.env.EXPO_PUBLIC_API_URL;
// export const API_URL = __DEV__
//   ? process.env.EXPO_PUBLIC_DEV_API_URL
//   : process.env.EXPO_PUBLIC_API_URL;
export const API_VERSION = process.env.EXPO_PUBLIC_API_VERSION;
export const API_BASE_URL = `${API_URL}/${API_VERSION}`;

const DOMAIN = {
  USER: 'users',
  AUTH: 'auth',
  STOCK: 'stock',
  NEWS: 'news',
};

/** Memo: API 명세서 도메인 및 endpoint 구조 적용 */
export const API_PATH = {
  USER: {
    INFO: DOMAIN.USER,
    CHECK_EMAIL: `${DOMAIN.USER}/check-email`,
    SIGN_IN: DOMAIN.USER,
    WITHDRAW: DOMAIN.USER,
    NEW: `${DOMAIN.USER}/new`,
  },
  AUTH: {
    SOCIAL: `${DOMAIN.AUTH}/social-login`,
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
    DETAIL: `${DOMAIN.NEWS}`,
    SCRAP: `${DOMAIN.NEWS}/scrap`,
    SCRAP_ADD: `${DOMAIN.NEWS}/scrap`,
    SCRAP_DELETE: `${DOMAIN.NEWS}/scrap`,
  },
};
