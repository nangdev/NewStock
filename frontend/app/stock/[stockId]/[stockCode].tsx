import { Entypo } from '@expo/vector-icons';
import { IMessage } from '@stomp/stompjs';
import { useAllStockNewsListQuery } from 'api/news/query';
import { useStockDetailInfoQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import SortButton from 'components/news/SortButton';
import NewsListItem from 'components/stock/NewsListItem';
import StockInfoCard from 'components/stock/StockInfoCard';
import { useLocalSearchParams } from 'expo-router';
import { useState, useEffect } from 'react';
import { View, ScrollView, Text, Image, Touchable, TouchableOpacity } from 'react-native';
import stompService from 'utils/stompService';

export default function StockDetail() {
  const { stockId, stockCode } = useLocalSearchParams();
  const id = Number(Array.isArray(stockId) ? stockId[0] : (stockId ?? 1));
  const code = Array.isArray(stockCode) ? stockCode[0] : (stockCode ?? '');

  const COUNT_PER_PAGE = 8;
  const [page, setPage] = useState<number>(0);
  const [sort, setSort] = useState<'score' | 'time'>('score');

  const [price, setPrice] = useState<number | null>(null);
  const [changeRate, setChangeRate] = useState<number | null>(null);

  const { data } = useStockDetailInfoQuery(id);
  const { data: newsListData } = useAllStockNewsListQuery(id, page, COUNT_PER_PAGE, sort);

  useEffect(() => {
    stompService.unsubscribe(code);
    stompService.connect();

    const onMessage = (msg: IMessage) => {
      const parsed = JSON.parse(msg.body);
      setPrice(parsed.price);
      setChangeRate(parsed.changeRate);
    };

    if (stompService.isReady()) {
      stompService.subscribe(code, onMessage);
    }

    return () => {
      stompService.unsubscribe(code);
    };
  }, [data]);

  const onPressLeft = () => {
    if (page > 0) {
      setPage((prev) => prev - 1);
    }
  };
  const onPressRight = () => {
    if (newsListData && page < newsListData?.data.totalPage - 1) {
      setPage((prev) => prev + 1);
    }
  };

  return (
    <>
      <CustomHeader title={data?.data.stockName} />
      <View className="mt-24">
        <StockInfoCard
          stockId={id}
          stockName={data?.data.stockName ?? ''}
          stockCode={code}
          price={price ?? data?.data.closingPrice ?? 0}
          changeRate={Number(changeRate ?? data?.data.rcPdcp ?? 0)}
          imgUrl={data?.data.stockImage ?? ''}
          totalPrice={data?.data.totalPrice ?? ''}
          issuedNum={data?.data.lstgStqt ?? ''}
          capital={data?.data.capital ?? ''}
          parValue={Number(data?.data.parValue ?? 0)}
          listingDate={data?.data.listingDate ?? ''}
          industry={data?.data.stdIccn ?? ''}
          priceChanged={data?.data.ctpdPrice ?? 0}
        />
      </View>

      <View className="my-4 flex-row items-center justify-between px-8">
        <Text className="ml-2 mt-1 text-xl font-semibold">관련 뉴스</Text>
        <View className="mt-1">
          <SortButton sort={sort} setSort={setSort} />
        </View>
      </View>

      <View className="mx-8 my-2 h-[405px] rounded-2xl bg-white pt-2 shadow-md">
        {newsListData && newsListData.data.newsList.length > 0 ? (
          <>
            <ScrollView>
              {newsListData?.data.newsList.map((news, index) => {
                const isLast = index === newsListData?.data.newsList.length - 1;
                return (
                  <NewsListItem
                    key={news.newsId}
                    newsId={news.newsId}
                    title={news.title}
                    description=""
                    score={news.score}
                    publishedDate={news.publishedDate}
                    isLast={isLast}
                  />
                );
              })}
            </ScrollView>
            <View className="mb-2 mt-4 flex-row items-center justify-center gap-4">
              {page > 0 ? (
                <TouchableOpacity
                  onPress={onPressLeft}
                  hitSlop={{ top: 10, bottom: 10, left: 10, right: 2 }}>
                  <Entypo name="triangle-left" size={18} />
                </TouchableOpacity>
              ) : (
                <Entypo name="triangle-left" size={18} color="#C7C7C7" />
              )}

              {newsListData && page < newsListData?.data.totalPage - 1 ? (
                <TouchableOpacity
                  onPress={onPressRight}
                  hitSlop={{ top: 10, bottom: 10, left: 2, right: 10 }}>
                  <Entypo name="triangle-right" size={18} />
                </TouchableOpacity>
              ) : (
                <Entypo name="triangle-right" size={18} color="#C7C7C7" />
              )}
            </View>
          </>
        ) : (
          <View className="flex-1 items-center justify-center">
            <Image
              source={require('../../../assets/image/no_data.png')}
              style={{ width: 50, height: 50, resizeMode: 'contain' }}
            />
            <Text style={{ color: '#8A96A3' }}>관련 뉴스가 없어요</Text>
          </View>
        )}
      </View>
      <CustomFooter />
    </>
  );
}
