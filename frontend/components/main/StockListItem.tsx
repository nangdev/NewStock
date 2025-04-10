import { AntDesign } from '@expo/vector-icons';
import { useTopFiveNewsListQuery } from 'api/news/query';
import NewsListItem from 'components/stock/NewsListItem';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import Collapsible from 'react-native-collapsible';

type StockProps = {
  stockId: number;
  stockName: string;
  stockCode: string;
  price: number;
  changeRate: number;
  imgUrl: string;
};

function StockListItem({ stockId, stockName, stockCode, price, changeRate, imgUrl }: StockProps) {
  const router = useRouter();
  const onPressItem = () => {
    router.navigate({
      pathname: ROUTE.STOCK.DETAIL(stockId, stockCode),
      params: { stockId, stockCode },
    });
  };

  const [expanded, setExpanded] = useState(false);
  const { data } = useTopFiveNewsListQuery(stockId);

  return (
    <>
      <TouchableOpacity onPress={onPressItem}>
        <View className="mx-8 my-2 overflow-hidden rounded-2xl bg-white shadow-md">
          <View className="flex-row items-center p-4">
            <Image
              source={{ uri: `data:image/png;base64,${imgUrl}` }}
              className="mr-6 h-16 w-16 rounded-xl bg-gray-200"
            />
            <View className="flex-1">
              <Text className="text-base font-bold">{stockName}</Text>
              <Text className="text-xs text-gray-500">{stockCode}</Text>
            </View>
            <View className="ml-auto mr-2 flex items-end">
              <Text className="text-base font-bold">{price.toLocaleString()} 원</Text>
              <Text
                className={`mt-1 text-sm ${
                  changeRate > 0
                    ? 'text-red-500'
                    : changeRate === 0
                      ? 'text-black'
                      : 'text-blue-500'
                }`}>
                {changeRate.toFixed(2)}%
              </Text>
            </View>
            <TouchableOpacity
              onPress={() => setExpanded(!expanded)}
              hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
              className="ml-2 p-1">
              <AntDesign name={expanded ? 'up' : 'down'} size={16} color="#888" />
            </TouchableOpacity>
          </View>
          <View className="mx-4 h-[1px] bg-gray-200" />

          <Collapsible collapsed={!expanded}>
            <View className="mb-2 items-center bg-white">
              {data?.data.newsList && data.data.newsList.length > 0 ? (
                data.data.newsList.map((news, idx) => {
                  const isLast = idx === data.data.newsList.length - 1;
                  return (
                    <NewsListItem
                      key={news.newsId}
                      newsId={news.newsId}
                      title={news.title}
                      description="desc"
                      score={news.score}
                      publishedDate={news.publishedDate}
                      isLast={isLast}
                    />
                  );
                })
              ) : (
                <View className="mb-5 items-center">
                  <Image
                    source={require('assets/image/no_data.png')}
                    style={{ width: 50, height: 50, resizeMode: 'contain' }}
                  />
                  <Text className="text-xs text-gray-500">오늘 관련 뉴스가 없어요</Text>
                </View>
              )}
            </View>
          </Collapsible>
        </View>
      </TouchableOpacity>
    </>
  );
}

export default React.memo(StockListItem);
