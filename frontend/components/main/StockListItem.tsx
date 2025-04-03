import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { AntDesign } from '@expo/vector-icons';
import Collapsible from 'react-native-collapsible';
import { useRouter } from 'expo-router';
import { ROUTE } from 'constants/routes';

type Props = {
  stockId: number;
  stockName: string;
  stockCode: string;
  price: number;
  changeRate: number;
  imgUrl: string;
  hojaeIconUrl: string;
};

 function StockListItem({
  stockId,
  stockName,
  stockCode,
  price,
  changeRate,
  imgUrl,
  hojaeIconUrl,
}: Props) {
  const router = useRouter();
  const onPressItem = () => {
    router.navigate({
      pathname: ROUTE.STOCK.DETAIL(stockId),
      params: {stockId}
    })
  }

  const [expanded, setExpanded] = useState(false);

  // mock data
  const newsList = [
    { id: `${stockCode}-1`, title: `${stockName} 관련 뉴스 1`, time: '1시간 전' },
    { id: `${stockCode}-2`, title: `${stockName} 관련 뉴스 2`, time: '2시간 전' },
    { id: `${stockCode}-3`, title: `${stockName} 관련 뉴스 3`, time: '2시간 전' },
    { id: `${stockCode}-4`, title: `${stockName} 관련 뉴스 4`, time: '2시간 전' },
    { id: `${stockCode}-5`, title: `${stockName} 관련 뉴스 5`, time: '2시간 전' },
  ];

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
          <View className="bg-white px-4 py-2 mb-2">
            {newsList.map((news) => (
              <View key={news.id} className="flex-row items-center my-1">
                <Image
                  source={{ uri: hojaeIconUrl || 'https://via.placeholder.com/36' }}
                  className="w-9 h-9 rounded-md mr-4 bg-gray-200"
                />
                <View className="flex-1 flex-row justify-between items-center">
                  <Text className="text-sm">{news.title}</Text>
                  <Text className="text-xs text-gray-500">{news.time}</Text>
                </View>
              </View>
            ))}
          </View>
        </Collapsible>
      </View>
    </TouchableOpacity>
  );
}

export default React.memo(StockListItem);