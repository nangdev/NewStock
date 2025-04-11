import { IMessage } from '@stomp/stompjs';
import { useAllUserStockListQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import StockListItem from 'components/main/StockListItem';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Text, View, ScrollView, ActivityIndicator, TouchableOpacity, Image } from 'react-native';
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
  const { data, isSuccess, isLoading } = useAllUserStockListQuery();
  const [subscribedStocks, setSubscribedStocks] = useState<Stock[]>([]);
  const userInfo = useUserStore();

  const onPressAddStock = () => {
    router.navigate(ROUTE.SET_INTEREST);
  };

  useEffect(() => {
    stompService.connect();

    if (!data?.data.stockList || !isSuccess) {
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
      <CustomHeader title="NewStock" disabled />
      {isLoading ? (
        <View className="h-full w-full items-center justify-center">
          <ActivityIndicator size="large" color="#724EDB" />
        </View>
      ) : (
        <>
          {data?.data.stockList.length ? (
            <View className="flex-1 py-6">
              <Text className="mb-4 ml-4 items-center px-6 text-lg font-semibold">
                <Text className="font-bold text-primary">{userInfo.userInfo?.nickname}</Text>
                님의 종목이에요
              </Text>

              <ScrollView className="gap-4">
                {subscribedStocks.map((stock) => (
                  <StockListItem
                    key={`${stock.stockCode}`}
                    stockId={stock.stockId}
                    stockName={stock.stockName}
                    stockCode={stock.stockCode}
                    price={stock.closingPrice ? stock.closingPrice : 0}
                    changeRate={+stock.rcPdcp}
                    imgUrl={stock.imgUrl}
                  />
                ))}
              </ScrollView>
            </View>
          ) : (
            <View className="h-full w-full items-center justify-center gap-8">
              <View className="items-center">
                <Image
                  source={require('../assets/image/no_data.png')}
                  style={{ width: 50, height: 50, resizeMode: 'contain' }}
                />
                <Text style={{ color: '#8A96A3' }}>관심 종목이 없어요</Text>
              </View>
              <TouchableOpacity onPress={onPressAddStock} className="rounded-lg bg-primary p-4">
                <Text className="text-center text-lg font-bold text-white">관심 종목 추가</Text>
              </TouchableOpacity>
            </View>
          )}
        </>
      )}
      <CustomFooter />
    </>
  );
}
