import { useMutation, useQuery } from '@tanstack/react-query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';

import { getAllStockList, getAllUserStockList, putStockInterest } from '.';

export const useAllStockListQuery = () => {
  return useQuery({
    queryKey: ['stockList'],
    queryFn: getAllStockList,
  });
};

export const useStockInterestMutation = () => {
  const router = useRouter();

  return useMutation({
    mutationFn: putStockInterest,
    onSuccess: () => {
      router.navigate(ROUTE.HOME);
    },
    onError: (error) => {
      // Todo: 에러 처리
      console.error(error);
    },
  });
};

export const useAllUserStockListQuery = () => {
  return useQuery({
    queryKey: ['userStockList'],
    queryFn: getAllUserStockList,
  });
};
