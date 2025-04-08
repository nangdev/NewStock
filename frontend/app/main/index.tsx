import { IMessage } from '@stomp/stompjs';
import { useAllUserStockListQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import StockListItem from 'components/main/StockListItem';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Text, View, ScrollView } from 'react-native';
import useUserStore from 'store/user';
import stompService from 'utils/stompService';

type Stock = {
  stockId: number;
  stockCode: string;
  stockName: string;
  closingPrice: number;
  rcPdcp: number;
  imgUrl: string;
};

export default function Main() {
  const router = useRouter();
  const { data } = useAllUserStockListQuery();
  const [subscribedStocks, setSubscribedStocks] = useState<Stock[]>([]);
  const userInfo = useUserStore();

  useEffect(() => {
    if (!data?.data.stockList) {
      router.push(`/stock/1/005930`);
      return;
    }

    const latestStockList = data.data.stockList;
    setSubscribedStocks(latestStockList);

    const onMessage = (msg: IMessage) => {
      const parsed = JSON.parse(msg.body);
      setSubscribedStocks((prev) =>
        prev.map((s) =>
          s.stockCode === parsed.stockCode
            ? { ...s, closingPrice: parsed.price, rcPdcp: parsed.changeRate }
            : s
        )
      );
    };

    stompService.connect();

    if (stompService.isReady()) {
      stompService.unsubscribeAll();
      latestStockList.forEach((stock) => {
        stompService.subscribe(stock.stockCode, onMessage);
      });
    }

    return () => {
      stompService.unsubscribeAll();
    };
  }, [data]);

  return ( 
    <>
      <CustomHeader />
      <View className="gap-4 py-24 pb-28">
        <Text className="px-6 text-lg font-bold">{userInfo.userInfo?.nickname}님의 종목이에요</Text>
        <ScrollView>
          {subscribedStocks.map((stock) => (
            <StockListItem
              key={stock.stockId}
              stockId={stock.stockId}
              stockName={stock.stockName}
              stockCode={stock.stockCode}
              price={stock.closingPrice ? stock.closingPrice : 0}
              changeRate={Number(stock.rcPdcp ?? 0)}
              imgUrl={stock.imgUrl ?? ''}
              hojaeIconUrl=""
            />
          ))}
        </ScrollView>
      </View>
      <CustomFooter />
    </>
  );
}
