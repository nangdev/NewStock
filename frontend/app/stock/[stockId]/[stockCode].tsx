import { View, ScrollView, Text } from "react-native";
import { useLocalSearchParams } from "expo-router";
import StockInfoCard from "components/stock/StockInfoCard";
import { useStockDetailInfoQuery } from "api/stock/query";
import { useAllStockNewsListQuery } from "api/news/query";
import NewsListItem from "components/stock/NewsListItem";
import { useState } from "react";

export default function StockDetail () {
  const { stockId, stockCode } = useLocalSearchParams();
  const [page, setPage] = useState<number>(1);
  const [count, setCount] = useState<number>(10);
  const [sort, setSort] = useState<'score' | 'time'>('score');

  const id = Number(Array.isArray(stockId) ? stockId[0] : stockId ?? 1);
  const code = Array.isArray(stockCode) ? stockCode[0] : stockCode ?? '';
  
  const { data, isLoading, isError } = useStockDetailInfoQuery(id);
  const { data: newsListData, isLoading: isNewsLoading, isError: isNewsError } = useAllStockNewsListQuery(
    id,
    page,
    count,
    sort,
  );

  type News = {
    newsId: string;
    title: string;
    description: string;
    score: number;
    publishedDate: string;
  }

  const newsList = newsListData?.data.newsList;

  return (
    <View className="">
      <Text>stock detail page {id} {code}</Text>
      <View className="mt-16">
        <StockInfoCard 
          stockId={id}
          stockName={data?.data.stockName ?? ''}
          stockCode={code}
          price={data?.data.closingPrice ?? 0}
          changeRate={Number(data?.data.rcPdcp ?? 0)}
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
        <Text>최신순</Text>
      </View>

      <View className="bg-white rounded-2xl mx-8 my-2 shadow-md max-h-[400px]">
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
        </ScrollView>
      </View>
    </View>
  );
}

