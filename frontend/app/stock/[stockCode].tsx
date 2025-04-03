import { View, ScrollView, Text } from "react-native";
import { useLocalSearchParams } from "expo-router";
import StockInfoCard from "components/stock/StockInfoCard";
import { useStockDetailInfoQuery } from "api/stock/query";
import NewsListItem from "components/stock/NewsListItem";
import { useState } from "react";


export default function StockDetail () {
  const { stockCode } = useLocalSearchParams();
  const code = Array.isArray(stockCode) ? stockCode[0] : stockCode ?? '';
  const { data, isLoading, isError } = useStockDetailInfoQuery(code);
  // mock data
  const newsList = [
    { id: `${stockCode}-1`, title: `관련 뉴스 1`, time: '1시간 전' },
    { id: `${stockCode}-2`, title: `관련 뉴스 2`, time: '2시간 전' },
    { id: `${stockCode}-3`, title: `관련 뉴스 3`, time: '2시간 전' },
    { id: `${stockCode}-4`, title: `관련 뉴스 4`, time: '2시간 전' },
    { id: `${stockCode}-5`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-6`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-7`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-8`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-9`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-10`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-11`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-12`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-13`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-14`, title: `관련 뉴스 5`, time: '2시간 전' },
    { id: `${stockCode}-15`, title: `관련 뉴스 5`, time: '2시간 전' },
  ];
  return (
    <View className="">
      <Text>stock detail page {code}</Text>
      <View className="mt-16">
        <StockInfoCard 
          // stockId={stockCode}
          stockName={data?.data.stockName ?? ''}
          stockCode={code}
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

