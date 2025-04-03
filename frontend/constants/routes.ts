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
    DETAIL: (stockId: number) => `/stock/${stockId}`,
  },
};
