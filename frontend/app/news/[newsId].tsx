import { AntDesign } from '@expo/vector-icons';
import {
  useNewsDetailQuery,
  useAddNewsScrapMutation,
  useDeleteNewsScrapMutation,
} from 'api/news/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
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

  const isHojae = newsInfo?.score > 0;

  const onPressPinIcon = () => {
    setIsScraped(!isScraped);
    isScraped ? deleteNewsScrap(id) : addNewsScrap(id);
  };

  return (
    <>
      <CustomHeader title={newsInfo.press} />
      <View className="mx-8 my-24 h-[680px] rounded-2xl bg-white p-6 shadow-md">
        <ScrollView>
          <View>
            <Image
              source={{ uri: newsInfo?.pressLogo }}
              className="h-6 w-14"
              resizeMode="contain"
            />

            <View className="flex-row items-center justify-between">
              <Text className="text-xl font-bold">{newsInfo?.title}</Text>
            </View>

            <View className="flex-row items-center justify-between">
              <Text className="my-4 text-gray-500">{newsInfo?.publishedDate}</Text>
              <AntDesign
                name={isScraped ? 'pushpin' : 'pushpino'}
                size={24}
                style={{ transform: [{ scaleX: -1 }] }}
                color="#724EDB"
                onPress={onPressPinIcon}
              />
            </View>
            <View className="border-t border-gray-200 " />
          </View>

          <Text className="m-2 text-lg">
            <Text className="font-bold">AI</Text>가 이 기사를{' '}
            <Text className={isHojae ? 'font-bold text-red-500' : 'font-bold text-blue-500'}>
              {isHojae ? '호재' : '악재'}
            </Text>
            로 분류했어요.
          </Text>

          <View className="my-4 ml-2 mr-12">
            <View className="mb-2 flex-row">
              <View className="mr-6 w-[2px] rounded-sm bg-black" />
              <Text className="mt-2 text-base">{newsInfo.newsSummary}</Text>
            </View>
          </View>

          {newsInfo.newsImage ? (
            <View className="px-2">
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
          <View className="mb-2 mt-8 border-t border-gray-200" />
          <Text className="text-sm text-gray-500">
            해당 기사의 저작권은 {newsInfo.press}에 었으며, 자세한 내용은 원문 링크를 통해 확인할 수
            있습니다.
          </Text>
          <Text className="text-sm text-gray-500">{newsInfo.url}</Text>
        </ScrollView>
      </View>
      <CustomFooter />
    </>
  );
}
