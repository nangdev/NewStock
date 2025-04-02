/** Memo: 0일 경우 신규 유저 */
export type UserInfoType = {
  userId: number;
  email: string;
  nickname: string;
  role: 0 | 1;
};
