import { ROUTE } from 'constants/routes';
import { getAllStockNewsList } from '.';
import { useQuery } from '@tanstack/react-query';


export const useAllStockNewsListQuery = (stockCode: string) => {
  return useQuery({
    queryKey: ['stockNewsList'],
    queryFn: () => getAllStockNewsList(stockCode),
  });    
}