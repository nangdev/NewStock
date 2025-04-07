import { useQuery } from '@tanstack/react-query';

import { getAllStockNewsList, getTopFiveStockNewsList } from '.';

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
