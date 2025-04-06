import { Text, View, ScrollView } from 'react-native';
import StockListItem from 'components/main/StockListItem';
import { Client, IMessage } from '@stomp/stompjs';
import { useEffect, useState } from 'react';
import { useAllUserStockListQuery } from 'api/stock/query';
import { useRouter } from 'expo-router';
import stompService from 'utils/stompService'

type Stock = {
  stockId: number,
  stockCode: string,
  stockName: string,
  closingPrice: number,
  rcPdcp: number,
  imgUrl: string,
}



export default function Main() {
  const router = useRouter();
  const { data, isLoading, isError } = useAllUserStockListQuery();
  const [subscribedStocks, setSubscribedStocks] = useState<Stock[]>([]);
  
  useEffect(() => {
    if (!data?.data.stockList) {
      router.replace(`/stock/1/005930`)
      return;
    }
    
    const latestStockList = data.data.stockList;
    setSubscribedStocks(latestStockList);

    const onMessage = (msg: IMessage) => {
      const parsed = JSON.parse(msg.body);
      // console.log(parsed)
      setSubscribedStocks((prev) =>
        prev.map((s) =>
          s.stockCode === parsed.stockCode
            ? { ...s, closingPrice: parsed.price, rcPdcp: parsed.changeRate }
            : s
        )
      );
    }

    
    stompService.connect();

    if (stompService.isReady()) {
      stompService.unsubscribeAll();
      latestStockList.forEach((stock) => {
        stompService.subscribe(stock.stockCode, onMessage);
      })
    }

    return () => {
      stompService.unsubscribeAll();
    }
  }, [data]);

  return (
    <View>
        <ScrollView>
          {subscribedStocks.map((stock) => (
            <StockListItem
              key={stock.stockCode}
              stockId={stock.stockId}
              stockName={stock.stockName}
              stockCode={stock.stockCode}
              price={stock.closingPrice ? stock.closingPrice : 0}
              changeRate={Number(stock.rcPdcp ?? 0)}
              imgUrl={stock.imgUrl ?? ''}
              hojaeIconUrl=''
          />
          ))}
        </ScrollView>
    </View>
  );
}
