import { Text, View, ScrollView } from 'react-native';
import StockListItem from 'components/main/StockListItem';
import { Client } from '@stomp/stompjs';
import { useEffect, useState } from 'react';
import { useAllUserStockListQuery } from 'api/stock/query';
import { useRouter } from 'expo-router';

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
    setSubscribedStocks(data.data.stockList);
    console.log(subscribedStocks)
    // STOMP 연결
    const client = new Client({
      webSocketFactory: () => new WebSocket('ws://j12a304.p.ssafy.io:8080/api/ws'),
      // webSocketFactory: () => new WebSocket('ws://10.0.2.2:8080/api/ws'),
      // debug: (msg) => console.log('STOMP:', msg),
      onConnect: (frame) => {
        // 구독
        subscribedStocks.forEach((stock) => {
          client.subscribe(`/topic/rtp/${stock.stockCode}`, (msg) => {
            const parsed = JSON.parse(msg.body);
            console.log(parsed)
            setSubscribedStocks((prev) =>
              prev.map((s) =>
                s.stockCode === parsed.stockCode
                  ? { ...s, closingPrice: parsed.price, rcPdcp: parsed.changeRate }
                  : s
              )
            );
          })
        })
      },
      forceBinaryWSFrames: true,
      appendMissingNULLonIncoming: true,
    })
    
    client.activate();

    return () => {
      client.deactivate();
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
              imgUrl={stock.imgUrl}
              hojaeIconUrl=''
          />
          ))}
        </ScrollView>
    </View>
  );
}
