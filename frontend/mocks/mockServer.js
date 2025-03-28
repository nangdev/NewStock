import { API_BASE_URL, API_PATH } from 'constants/api';
import { createServer } from 'miragejs';

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

      // Memo: 회원가입
      this.post(`${API_BASE_URL}/${API_PATH.USER.SIGN_IN}`, () => {
        return {};
      });

      // Memo: 이메일 중복 검사
      this.get(`${API_BASE_URL}/${API_PATH.USER.CHECK_EMAIL}`, () => {
        return {
          data: {
            isDuplicated: true,
          },
        };
      });
    },
  });
}
