import { api } from 'api/axiosInstance';
import { API_PATH } from 'constants/api';
import { AllStockNewsResType } from 'types/api/news';

export const getAllStockNewsList = async (stockCode: string) => {
  const response = await api.get<AllStockNewsResType>(API_PATH.NEWS.STOCK_NEWS);
  return response.data;
}