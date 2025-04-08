import { useMutation, useQuery } from '@tanstack/react-query';

import {
  deleteNewsScrap,
  getAllStockNewsList,
  getNewsScrapList,
  getNewsDetailInfo,
  getTopFiveStockNewsList,
  postNewsScrap,
} from '.';

export const useAllStockNewsListQuery = (
  stockId: number,
  page: number,
  count: number,
  sort: 'score' | 'time'
) => {
  return useQuery({
    queryKey: ['stockNewsList', stockId, page, count, sort],
    queryFn: () => getAllStockNewsList(stockId, page, count, sort),
  });
};

export const useTopFiveNewsListQuery = (stockId: number) => {
  return useQuery({
    queryKey: ['topFiveNewsList', stockId],
    queryFn: () => getTopFiveStockNewsList(stockId),
  });
};

export const useNewsScrapListQuery = ({
  stockCode,
  page,
  count,
  sort,
}: {
  stockCode: string;
  page: number;
  count: number;
  sort: string;
}) => {
  return useQuery({
    queryKey: ['newsScrapList'],
    queryFn: () => getNewsScrapList(stockCode, page, count, sort),
  });
};

export const useAddNewsScrapMutation = () => {
  return useMutation({
    mutationFn: postNewsScrap,
    onSuccess: () => {
      console.log('뉴스 스크랩 추가 성공');
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};

export const useDeleteNewsScrapMutation = () => {
  return useMutation({
    mutationFn: deleteNewsScrap,
    onSuccess: () => {
      console.log('뉴스 스크랩 삭제 성공');
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};

export const useNewsDetailQuery = (newsId: number) => {
  return useQuery({
    queryKey: ['newsDetail', newsId],
    queryFn: () => getNewsDetailInfo(newsId),
  });
};
