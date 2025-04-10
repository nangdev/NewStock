import { AntDesign } from '@expo/vector-icons';
import {
  useNewsDetailQuery,
  useAddNewsScrapMutation,
  useDeleteNewsScrapMutation,
} from 'api/news/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import ScoreBoard from 'components/news/ScoreBoard';
import { useLocalSearchParams } from 'expo-router';
import { useState, useEffect } from 'react';
import { View, Text, ScrollView, Image, ActivityIndicator } from 'react-native';

export default function NewsDetailPage() {
  const { newsId } = useLocalSearchParams();
  const id = Number(Array.isArray(newsId) ? newsId[0] : (newsId ?? 1));
  const { data, isLoading, isError } = useNewsDetailQuery(id);
  const { mutate: addNewsScrap } = useAddNewsScrapMutation();
  const { mutate: deleteNewsScrap } = useDeleteNewsScrapMutation();

  const [isScraped, setIsScraped] = useState(false);

  useEffect(() => {
    if (data?.data?.isScraped !== undefined) {
      setIsScraped(data.data.isScraped);
    }
  }, [data?.data.isScraped]);

  if (isLoading) {
    return (
      <View className="h-full w-full items-center justify-center">
        <ActivityIndicator size="large" color="#724EDB" />
      </View>
    );
  }

  if (isError || !data?.data.newsInfo) {
    return (
      <>
        <CustomHeader />
        <CustomFooter />
      </>
    );
  }

  const newsInfo = { ...data?.data.newsInfo };

  const sentimentValue = newsInfo.score >= 2.5 ? '호재' : newsInfo.score <= -2.5 ? '악재' : '중립';
  const sentimentColor = newsInfo.score >= 2.5 ? 'text-red-500' : newsInfo.score <= -2.5 ? 'text-blue-500' : '';

  const onPressPinIcon = () => {
    setIsScraped(!isScraped);
    isScraped ? deleteNewsScrap(id) : addNewsScrap(id);
  };

  return (
    <>
      <CustomHeader title={newsInfo.press} />
      <View className="mx-4 my-24 h-[680px] rounded-2xl bg-white p-6 shadow-md">
        <ScrollView
        	showsVerticalScrollIndicator={false}
        >
          <View>
          <View className="flex-row justify-between items-center mb-2">
            <Image
              source={{ uri: newsInfo?.pressLogo }}
              className="h-6 w-14"
              resizeMode="contain"
            />
            <AntDesign
              name={isScraped ? 'star' : 'staro'}
              size={16}
              color="#724EDB"
              onPress={onPressPinIcon}
              style={{ transform: [{ scaleX: -1 }] }}
            />
          </View>

            <View className="flex-row items-center justify-between">
              <Text className="flex-1 text-xl font-bold" style={{textAlign: 'justify'}}>{newsInfo?.title}</Text>
            </View>

            <View className="flex-row items-center justify-between">
              <Text className="my-4 mt-2 text-gray-500">{newsInfo?.publishedDate}</Text>
            </View>
            <View className="border-t border-gray-200 " />
          </View>
          
          <View className='flex-1 mt-2 items-center justify-center'>
            <Text className="mt-2 text-lg">
              <Text className="font-bold">AI</Text>가 이 기사를{' '}
              <Text className={`font-bold ${sentimentColor}`}>
                {sentimentValue}
              </Text>
              {sentimentValue==='중립' ? '으' : ''}로 분류했어요.
            </Text>
          </View>

          <ScoreBoard
            score={newsInfo.score}
            sentimentColor={sentimentColor}
            financeScore={newsInfo.financeScore}
            strategyScore={newsInfo.strategyScore}
            governScore={newsInfo.governScore}
            techScore={newsInfo.techScore}
            externalScore={newsInfo.externalScore}
          />

          <View className="p-4">
            <View className="flex-row justify-center">
              <View className="mr-4 w-[2px] rounded-sm bg-black" />
              <Text className="text-base">{newsInfo.newsSummary}</Text>
            </View>
          </View>

          {newsInfo.newsImage ? (
            <View className="mx-2 mb-8 mt-4">
              <Image
                source={{ uri: newsInfo.newsImage }}
                className="h-60 w-full rounded-lg"
                resizeMode="contain"
              />
            </View>
          ) : null}

          <View>
            <Text className="text-lg">{newsInfo.content}</Text>
          </View>
          <View className="mb-2 mt-4 border-t border-gray-200" />
          <Text className="text-sm text-gray-500 mb-2">
            해당 기사의 저작권은 {newsInfo.press}에 있으며, 자세한 내용은 원문 링크를 통해 확인할 수
            있습니다.
          </Text>
          <Text className="text-sm text-gray-500">원문 링크: {newsInfo.url}</Text>
        </ScrollView>
      </View>
      <CustomFooter />
    </>
  );
}
