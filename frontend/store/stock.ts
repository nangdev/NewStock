import { StockType } from 'types/api/stock';
import { create } from 'zustand';

type StockStoreType = {
  stockList: StockType[] | [];
  interestedStockList: StockType[] | [];
  setStockList: (stockList: StockType[]) => void;
  setInterestedStockList: (interestedStockList: StockType[]) => void;
};

const useStockStore = create<StockStoreType>((set) => ({
  stockList: [],
  interestedStockList: [],
  setStockList: (stockList) => set({ stockList }),
  setInterestedStockList: (interestedStockList) => set({ interestedStockList }),
}));

export default useStockStore;
