import { API_BASE_URL, API_PATH } from 'constants/api';
import { createServer, Response } from 'miragejs';

import {
  mockAllStock,
  mockNewsletter,
  mockNotificationList,
  mockScrapNewsList,
  mockUserStock,
} from './mockDataBase';

export function makeServer({ environment = 'development' } = {}) {
  if (window.server) {
    console.log('MirageJS 서버가 이미 실행 중입니다.');
    return window.server;
  }

  return createServer({
    environment,

    routes() {
      // Memo: 로그인
      this.post(`${API_BASE_URL}/${API_PATH.AUTH.LOGIN}`, () => {
        return {
          data: {
            accessToken: 'accessToken123',
            refreshToken: 'refreshToken123',
          },
        };
      });

      // Memo: 소셜 로그인
      this.post(`${API_BASE_URL}/${API_PATH.AUTH.SOCIAL}`, () => {
        return {
          data: {
            accessToken: 'accessToken123',
            refreshToken: 'refreshToken123',
          },
        };
      });

      // Memo: 로그아웃
      this.post(`${API_BASE_URL}/${API_PATH.AUTH.LOGOUT}`, () => {
        return {};
      });

      // Memo: 리프레쉬 요청
      this.post(`${API_BASE_URL}/${API_PATH.AUTH.REFRESH}`, () => {
        return {
          accessToken: 'access123123',
          refreshToken: 'refresh123123',
        };
      });

      // Memo: 회원가입
      this.post(`${API_BASE_URL}/${API_PATH.USER.SIGN_UP}`, () => {
        return {};
      });

      // Memo: 이메일 중복 검사
      this.get(`${API_BASE_URL}/${API_PATH.USER.CHECK_EMAIL}`, () => {
        return {
          data: {
            isDuplicated: false,
          },
        };
      });

      // Memo: 유저 정보 조회
      this.get(`${API_BASE_URL}/${API_PATH.USER.INFO}`, () => {
        return {
          data: {
            userId: '1',
            email: 'ssafy@naver.com',
            nickname: '싸피좋아',
            role: 0,
          },
        };
      });

      // Memo: 최초 로그인 시 유저 권한 변경
      this.put(`${API_BASE_URL}/${API_PATH.USER.NEW}`, () => {
        return {};
      });

      // Memo: 전체 주식 정보 조회
      this.get(`${API_BASE_URL}/${API_PATH.STOCK.ALL}`, () => {
        return mockAllStock;
      });

      // Memo: 회원 관심 주식 조회
      this.get(`${API_BASE_URL}/${API_PATH.STOCK.USER_STOCK}`, () => {
        return mockUserStock;
      });

      // Memo: 회원 관심 주식 수정
      this.put(`${API_BASE_URL}/${API_PATH.STOCK.USER_STOCK_EDIT}`, () => {
        return {};
      });

      // Memo: 알림 목록 조회
      this.get(`${API_BASE_URL}/${API_PATH.NOTIFICATION.ALL}`, () => {
        return mockNotificationList;
      });

      // Memo: 알림 읽음
      this.put(`${API_BASE_URL}/${API_PATH.NOTIFICATION.READ}/:unId`, () => {
        return {};
      });

      // Memo: 알림 삭제
      this.delete(`${API_BASE_URL}/${API_PATH.NOTIFICATION.DELETE}/:unId`, () => {
        return {};
      });

      // Memo: 뉴스 레터 조회
      this.get(`${API_BASE_URL}/${API_PATH.NEWSLETTER}/:date`, () => {
        return mockNewsletter;
      });

      // Memo: 스크랩 뉴스 조회
      this.get(`${API_BASE_URL}/${API_PATH.NEWS.SCRAP}`, () => {
        return mockScrapNewsList;
      });

      // Memo: 스크랩 뉴스 추가
      this.post(`${API_BASE_URL}/${API_PATH.NEWS.SCRAP_ADD}/:newsId`, () => {
        return {};
      });

      // Memo: 스크랩 뉴스 삭제
      this.delete(`${API_BASE_URL}/${API_PATH.NEWS.DELETE}/:newsId`, () => {
        return {};
      });
    },
  });
}
