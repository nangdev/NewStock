export const ROUTE = {
  HOME: '/',
  INTRO: {
    INTRO: '/intro',
    ONBOARDING: '/intro/onboarding',
  },
  USER: {
    SIGNUP: '/user/signup',
    LOGIN: '/user/login',
    OAUTH: '/user/kakaoLogin',
  },
  STOCK: {
    DETAIL: (stockId: number, stockCode: string) => `/stock/${stockId}/${stockCode}`,
  },
  NEWS: {
    STOCK_NEWS: (stockId: number) => `/news/${stockId}`,
    TOP: (stockId: number) => `/news/top/${stockId}`,
    DETAIL: (newsId: number) => `/news/${newsId}`,
    SCRAP: '/news/scrap',
    SCRAP_NEWS: (stockCode: string) => `/news/scrap/${stockCode}`,
    STOCK: '/news/stock',
  },
  MYPAGE: '/mypage',
  SET_INTEREST: '/mypage/setInterest',
  NEWSLETTER: {
    CALENDAR: '/newsletter',
    INDEX: '/newsletter',
  },
};
