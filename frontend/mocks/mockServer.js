import { API_BASE_URL, API_PATH } from 'constants/api';
import { createServer, Response } from 'miragejs';

import { mockAllStock } from './mockDataBase';

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
            role: 0,
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
      this.post(`${API_BASE_URL}/${API_PATH.USER.SIGN_IN}`, () => {
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
            userName: '김싸피',
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

      // Memo: 회원 관심 주식 수정
      this.put(`${API_BASE_URL}/${API_PATH.STOCK.USER_STOCK_EDIT}`, () => {
        return {};
      });
    },
  });
}
