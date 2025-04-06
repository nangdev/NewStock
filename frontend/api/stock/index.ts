import { api } from 'api/axiosInstance';
import { API_PATH } from 'constants/api';
import {
  AllUserStockResType,
  AllStockResType,
  StockInterestReqType,
  StockInterestResType,
} from 'types/api/stock';

export const getAllStockList = async () => {
  const response = await api.get<AllStockResType>(API_PATH.STOCK.ALL);
  return response.data;
};

export const putStockInterest = async ({ stockIdList }: StockInterestReqType) => {
  const response = await api.put<StockInterestResType>(API_PATH.STOCK.USER_STOCK_EDIT, {
    stockIdList,
  });
  return response.data;
};

export const getAllUserStockList = async () => {
  const response = await api.get<AllUserStockResType>(API_PATH.STOCK.USER_STOCK);
  return response.data;
};
