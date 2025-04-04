export const ROUTE = {
  HOME: '/',
  INTRO: {
    INTRO: '/intro',
    ONBOARDING: '/intro/onboarding',
  },
  USER: {
    SIGNUP: '/user/signup',
    LOGIN: '/user/login',
  },
  STOCK: {
    DETAIL: (stockId: number, stockCode: string) => `/stock/${stockId}/${stockCode}`,
  },
  NEWS: {
    STOCK_NEWS: (stockId: number) => `/news/${stockId}`,
    TOP: (stockId: number) => `/news/top/${stockId}`,
  },
};
