import { Entypo } from '@expo/vector-icons';
import { useNewsScrapListQuery } from 'api/news/query';
import { useAllUserStockListQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import SortButton from 'components/news/SortButton';
import { ROUTE } from 'constants/routes';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { View, Text, FlatList, TouchableOpacity, ActivityIndicator, Image } from 'react-native';
import { NewsType } from 'types/api/news';
import { getTimeAgo } from 'utils/date';

export default function StockNewsScrapPage() {
  const stockCode = useLocalSearchParams().stockCode.toString();
  const stockId = Number(useLocalSearchParams().stockId);
  const route = useRouter();
  const [page, setPage] = useState(0);
  const [sort, setSort] = useState<'score' | 'time'>('time');

  const COUNT_PER_PAGE = 10;

  
  const { data, isLoading, isError } = useNewsScrapListQuery({
    stockId,
    page,
    count: COUNT_PER_PAGE,
    sort,
  });
  const { data: userStockData } = useAllUserStockListQuery();
  
  if (isLoading) {
    return (
      <View className="h-full w-full items-center justify-center">
        <ActivityIndicator size="large" color="#724EDB" />
      </View>
    );
  }

  const stockName = userStockData?.data.stockList.find(
    (stock) => stock.stockCode === stockCode
  )?.stockName;

  if (isError || !data?.data.newsList.length) {
    return (
      <>
      <CustomHeader title={stockName ?? ''}/>
      <View className="h-full w-full items-center justify-center gap-8">
        <Image
          source={require('../../../assets/image/no_data.png')}
          style={{ width: 50, height: 50, resizeMode: 'contain' }}
        />
        <Text style={{ color: '#8A96A3' }}>스크랩한 뉴스가 없어요</Text>
      </View>
      <CustomFooter/>
      </>
    );
  }

  const onPressLeft = () => {
    if (page > 0) {
      setPage((prev) => prev - 1);
    }
  };
  const onPressRight = () => {
    if (page < data.data.totalPage - 1) {
      setPage((prev) => prev + 1);
    }
  };

  const renderItem = ({ item }: { item: NewsType }) => (
    <TouchableOpacity
      onPress={() => route.navigate(ROUTE.NEWS.DETAIL(item.newsId))}
      className="border-b border-gray-200 px-4 py-3">
      <Text className="mb-1 text-base font-medium text-black" numberOfLines={1}>
        {item.title}
      </Text>
      <Text className="mb-1 text-sm font-medium text-black" numberOfLines={1}>
        {item.description}
      </Text>
      <Text className="self-end text-xs text-gray-400">{getTimeAgo(item.publishedDate)}</Text>
    </TouchableOpacity>
  );


  return (
    <>
      <CustomHeader title={stockName} />
      <View className="gap-4 px-4 py-24">
        <View className="flex-row justify-between">
          <Text className="text-xl font-bold">스크랩한 뉴스예요</Text>
          <SortButton sort={sort} setSort={setSort} />
        </View>
        <View className="h-[600px] w-full rounded-2xl bg-white shadow-lg">
          <FlatList
            data={data?.data.newsList}
            keyExtractor={(item) => String(item.newsId)}
            renderItem={renderItem}
          />
        </View>
        <View className="mb-2 mt-4 flex-row items-center justify-center gap-4">
          {page > 0 ? (
            <TouchableOpacity
              onPress={onPressLeft}
              hitSlop={{ top: 10, bottom: 10, left: 10, right: 2 }}
              >
              <Entypo name="triangle-left" size={18} />
            </TouchableOpacity>
          ) : (
              <Entypo name="triangle-left" size={18} color="#C7C7C7" />
          )}

          {page < data?.data.totalPage - 1 ? (
            <TouchableOpacity
              onPress={onPressRight}
              hitSlop={{ top: 10, bottom: 10, left: 2, right: 10 }}
            >
              <Entypo name="triangle-right" size={18} />
            </TouchableOpacity>
          ) : (
            <Entypo name="triangle-right" size={18} color="#C7C7C7" />
          )}
        </View>
        
      </View>
      <CustomFooter />
    </>
  );
}
