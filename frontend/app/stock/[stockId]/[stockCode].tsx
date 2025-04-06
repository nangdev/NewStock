import { View, ScrollView, Text } from "react-native";
import { useLocalSearchParams } from "expo-router";
import StockInfoCard from "components/stock/StockInfoCard";
import { useStockDetailInfoQuery } from "api/stock/query";
import { useAllStockNewsListQuery } from "api/news/query";
import NewsListItem from "components/stock/NewsListItem";
import SortButton from "components/news/SortButton";
import { useState, useEffect } from "react";
import stompService from "utils/stompService";
import { IMessage } from "@stomp/stompjs";
import { AntDesign } from "@expo/vector-icons";

type News = {
  newsId: string;
  title: string;
  description: string;
  score: number;
  publishedDate: string;
}

export default function StockDetail () {
  const { stockId, stockCode } = useLocalSearchParams();
  const id = Number(Array.isArray(stockId) ? stockId[0] : stockId ?? 1);
  const code = Array.isArray(stockCode) ? stockCode[0] : stockCode ?? '';

  const [page, setPage] = useState<number>(0);
  const [count, setCount] = useState<number>(6);
  const [sort, setSort] = useState<'score' | 'time'>('score');

  const [price, setPrice] = useState<number | null>(null);
  const [changeRate, setChangeRate] = useState<number | null>(null);
  
  const { data, isLoading, isError } = useStockDetailInfoQuery(id);
  const { data: newsListData, isLoading: isNewsLoading, isError: isNewsError } = useAllStockNewsListQuery(
    id,
    page,
    count,
    sort,
  );
  
  useEffect(() => {
    const newsList = newsListData?.data.newsList;

    stompService.unsubscribe(code);
    stompService.connect();

    const onMessage = (msg: IMessage) => {
      const parsed = JSON.parse(msg.body);
      setPrice(parsed.price);
      setChangeRate(parsed.changeRate);
    }

    if (stompService.isReady()) {
      stompService.subscribe(code, onMessage)
    }

    return () => {
      stompService.unsubscribe(code);
    }
  }, [data])

  const onPressLeft = () => {
    setPage(page-1);
  }
  const onPressRight = () => {
    setPage(page+1);
  }

  


  return (
    <View className="">
      <Text>stock detail page {id} {code}</Text>
      <View className="mt-16">
        <StockInfoCard 
          stockId={id}
          stockName={data?.data.stockName ?? ''}
          stockCode={code}
          price={price ?? data?.data.closingPrice ?? 0}
          changeRate={Number(changeRate ?? data?.data.rcPdcp ?? 0)}
          imgUrl={data?.data.stockImage ?? ''}
          totalPrice={Number(data?.data.totalPrice ?? 0)}
          issuedNum={Number(data?.data.lstgStqt ?? 0)}
          capital={Number(data?.data.capital ?? 0)}
          parValue={Number(data?.data.parValue ?? 0)}
          listingDate={data?.data.listingDate ?? ''}
          industry={data?.data.stdIccn ?? ''}          
        />
      </View>
      
      <View className="flex-row justify-between items-center px-8 my-8">
        <Text className="text-xl">관련 뉴스</Text>
        <SortButton sort={sort} setSort={setSort} />
      </View>

      <View className="bg-white rounded-2xl mx-8 my-2 shadow-md max-h-[400px] pt-2">
        <ScrollView>
          {newsListData?.data.newsList.map((news) => (
            <NewsListItem
              key={news.newsId}
              newsId={news.newsId}
              title={news.title}
              description={''}
              score={news.score}
              publishedDate={news.publishedDate}
              hojaeIconUrl={""}
            />
          ))}
          <View className="flex-row justify-center items-center mt-4 mb-2">
            <AntDesign name='left' onPress={onPressLeft}/>
            <Text className="mx-2">{page+1}</Text>
            <AntDesign name='right' onPress={onPressRight}/>
          </View>
        </ScrollView>
      </View>
    </View>
  );
}

