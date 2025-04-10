export const getTimeAgo = (dateString: string) => {
  const date = new Date(dateString);

  const nowUTC = new Date();
  const nowKST = new Date(nowUTC.getTime() + 9 * 60 * 60 * 1000);

  const diff = (nowKST.getTime() - date.getTime()) / 1000;

  if (diff < 60) return '방금 전';
  if (diff < 60 * 60) return `${Math.floor(diff / 60)}분 전`;
  if (diff < 60 * 60 * 24) return `${Math.floor(diff / 60 / 60)}시간 전`;
  if (diff < 60 * 60 * 24 * 7) return `${Math.floor(diff / 60 / 60 / 24)}일 전`;

  return date.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
};
