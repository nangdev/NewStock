import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { AntDesign } from '@expo/vector-icons';
import Collapsible from 'react-native-collapsible';
import { useRouter } from 'expo-router';
import { ROUTE } from 'constants/routes';
import NewsListItem from 'components/stock/NewsListItem';
import { useTopFiveNewsListQuery } from 'api/news/query';

type StockProps = {
  stockId: number;
  stockName: string;
  stockCode: string;
  price: number;
  changeRate: number;
  imgUrl: string;
  hojaeIconUrl: string;
};

type News = {
  newsId: number;
  title: string;
  description: string;
  score: number;
  publishedDate: string;
}

 function StockListItem({
  stockId,
  stockName,
  stockCode,
  price,
  changeRate,
  imgUrl,
  hojaeIconUrl,
}: StockProps) {
  const router = useRouter();
  const onPressItem = () => {
    router.navigate({
      pathname: ROUTE.STOCK.DETAIL(stockId, stockCode),
      params: {stockId, stockCode}
    })
  }

  const [expanded, setExpanded] = useState(false);
  const {data, isLoading, isError} = useTopFiveNewsListQuery(stockId);

  // mock data
  // const newsList = [
  //   { id: `${stockCode}-1`, title: `${stockName} 관련 뉴스 1`, time: '1시간 전' },
  //   { id: `${stockCode}-2`, title: `${stockName} 관련 뉴스 2`, time: '2시간 전' },
  //   { id: `${stockCode}-3`, title: `${stockName} 관련 뉴스 3`, time: '2시간 전' },
  //   { id: `${stockCode}-4`, title: `${stockName} 관련 뉴스 4`, time: '2시간 전' },
  //   { id: `${stockCode}-5`, title: `${stockName} 관련 뉴스 5`, time: '2시간 전' },
  // ];

  return (
    <TouchableOpacity onPress={onPressItem}>
      <View className="bg-white rounded-2xl mx-8 my-2 shadow-md overflow-hidden">
        <View className="flex-row items-center p-4">
          <Image
            source={{ uri: `data:image/png;base64,${imgUrl}` }}
            className="w-16 h-16 rounded-xl mr-6 bg-gray-200"
          />
          <View className="flex-1">
            <Text className="text-base font-bold">{stockName}</Text>
            <Text className="text-xs text-gray-500">{stockCode}</Text>
          </View>
          <View className="flex-1 items-end mr-6">
            <Text className="text-base font-bold">{price.toLocaleString()} 원</Text>
            <Text className={`text-sm mt-1 ${changeRate > 0
            ? 'text-red-500' 
            : changeRate == 0
            ? 'text-black'
            : 'text-blue-500'}`}>
            {changeRate.toFixed(2)}%
          </Text>
          </View>
          <TouchableOpacity onPress={() => setExpanded(!expanded)}>
            <AntDesign name={expanded ? 'up' : 'down'} size={14} color="#888" className="ml-2" />
          </TouchableOpacity>
        </View>

        <Collapsible collapsed={!expanded}>
          
          <View className="bg-white mb-2 items-center">
            {data?.data.newsList && data?.data.newsList.length > 0
            ? data?.data.newsList.map((news) => (
              <NewsListItem
                newsId={+news.newsId}
                title={news.title}
                description='desc'
                score={10}
                publishedDate='2025-03-14:20:08:49'
                hojaeIconUrl=''
              />
            ))
            : (
              <View className='items-center mb-5'>
                <Image
                  source={require('assets/image/no_data.png')}
                  style={{ width: 50, height: 50, resizeMode: 'contain' }}>
                </Image>
                <Text className='' style={{color: '#8A96A3'}}>오늘 관련 뉴스가 없어요</Text>
              </View>
            )
          }
          </View>
        </Collapsible>
      </View>
    </TouchableOpacity>
  );
}

export default React.memo(StockListItem);