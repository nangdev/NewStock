import { Text, View, ScrollView } from 'react-native';
import StockListItem from 'components/main/StockListItem';
import { Client } from '@stomp/stompjs';
import { useEffect, useState } from 'react';

export default function Main() {
  
  // 테스트 데이터
  const [subscribedStocks, setSubscribedStocks] = useState([
    { stockName: '삼성전자', stockCode: '005930', price: 0, changeRate: 0.0 },
    { stockName: 'SK하이닉스', stockCode: '000660', price: 0, changeRate: 0.0 },
    { stockName: '카카오', stockCode: '035720', price: 0, changeRate: 0.0 },
  ]);

  useEffect(() => {
    // STOMP 연결
    const client = new Client({
      webSocketFactory: () => new WebSocket('ws://10.0.2.2:8080/api/ws'),
      // debug: (msg) => console.log('STOMP:', msg),
      onConnect: (frame) => {
        // 구독
        subscribedStocks.forEach((stock) => {
          client.subscribe(`/topic/rtp/${stock.stockCode}`, (msg) => {
            const parsed = JSON.parse(msg.body);
            setSubscribedStocks((prev) =>
              prev.map((s) =>
                s.stockCode === parsed.stockCode
                  ? { ...s, price: parsed.price, changeRate: parsed.changeRate }
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
  }, []);

  return (
    <View>
      <ScrollView>
        {subscribedStocks.map((stock) => (
          <StockListItem
            stockName={stock.stockName}
            stockCode={stock.stockCode}
            price={stock.price}
            changeRate={stock.changeRate}
            imgUrl=''
            hojaeIconUrl=''
        />
        ))}
      </ScrollView>
    </View>
  );
}
