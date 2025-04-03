import { View, ScrollView, Text } from "react-native";
import { useLocalSearchParams } from "expo-router";
import StockInfoCard from "components/stock/StockInfoCard";
import { useStockDetailInfoQuery } from "api/stock/query";
import NewsListItem from "components/stock/NewsListItem";
import { useState } from "react";


export default function StockDetail () {
  const { stockId } = useLocalSearchParams();
  const code = Array.isArray(stockId) ? stockId[0] : stockId ?? '';
  const { data, isLoading, isError } = useStockDetailInfoQuery(+stockId);
  // mock data
  const newsList = [
    { id: `${stockId}-1`, title: `관련 뉴스 1`, time: '1시간 전' },
    { id: `${stockId}-2`, title: `관련 뉴스 2`, time: '2시간 전' },
    { id: `${stockId}-3`, title: `관련 뉴스 3`, time: '2시간 전' },
    { id: `${stockId}-4`, title: `관련 뉴스 4`, time: '2시간 전' },
    { id: `${stockId}-5`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-6`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-7`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-8`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-9`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-10`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-11`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-12`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-13`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-14`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockId}-15`, title: `관련 뉴스 5`, time: '2시간 전' },
  ];
  return (
    <View className="">
      <Text>stock detail page {stockId}</Text>
      <View className="mt-16">
        <StockInfoCard 
          stockId={+stockId}
          stockName={data?.data.stockName ?? ''}
          // stockCode={}
          price={data?.data.closingPrice ?? 0}
          changeRate={Number(data?.data.rcPdcp ?? 0)}
          imgUrl={data?.data.stockImage ?? ''}
        />
      </View>
      
      <View className="flex-row justify-between items-center px-8 my-8">
        <Text className="text-xl">관련 뉴스</Text>
        <Text>최신순</Text>
      </View>

      <View className="bg-white rounded-2xl mx-8 my-2 shadow-md max-h-[400px]">
        <ScrollView>
          {newsList.map((news) => (
            <NewsListItem
              key={news.id}
              newsId={news.id}
              title={news.title}
              description={"desc"}
              score={10}
              publishedDate={"2025-03-14:20:08:49"}
              hojaeIconUrl={""}
            />
          ))}
        </ScrollView>
      </View>
    </View>
  );
}

