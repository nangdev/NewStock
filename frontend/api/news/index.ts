import { api } from 'api/axiosInstance';
import { API_PATH } from 'constants/api';
import { AllStockNewsResType } from 'types/api/news';

export const getAllStockNewsList = async (
  stockId: number,
  page: number,
  count: number,
  sort: 'score' | 'time'
) => {
  const response = await api.get<AllStockNewsResType>(API_PATH.NEWS.STOCK_NEWS, {
    params: {
      stockId,
      page,
      count,
      sort,
    },
  });
  return response.data;
};

export const getTopFiveStockNewsList = async (stockId: number) => {
  const response = await api.get<AllStockNewsResType>(API_PATH.NEWS.TOP(stockId));
  return response.data;
};
