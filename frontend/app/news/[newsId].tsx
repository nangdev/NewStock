import { AntDesign, Fontisto } from '@expo/vector-icons';
import { useNewsDetailQuery, useAddNewsScrapMutation, useDeleteNewsScrapMutation } from 'api/news/query';
import { useLocalSearchParams } from 'expo-router';
import { useState, useEffect } from 'react';
import { View, Text, ScrollView, Image } from 'react-native';

export default function NewsDetailPage() {
  const { newsId } = useLocalSearchParams();
  const id = Number(Array.isArray(newsId) ? newsId[0] : (newsId ?? 1));
  const { data, isLoading, isError } = useNewsDetailQuery(id);
  const { mutate: addNewsScrap } = useAddNewsScrapMutation();
  const { mutate: deleteNewsScrap } = useDeleteNewsScrapMutation();

  // const title = "공개된 삼성전자 2025년형 'AI TV' [포토]";
  // const content = "삼성전자 언박스&디스커버 2025unbox & discover 2025 미디어데이가 7일 오전 서울 서초구 삼성 강남에서 열려 신제품들이 취재진에 공개되고 있다. 사진영상기획부 발로 뛰는 더팩트는 24시간 여러분의 제보를 기다립니다.  카카오톡: 더팩트제보 검색  이메일:   뉴스 홈페이지: http://talk.tf.co.kr/bbs/report/write";
  // const publishedDate = "2025-04-07 11:12:13";
  // const newsSummary = "7일 오전 서울 서초구 삼성 강남에서 열린 삼성전자 언박스&디스커버 2025unbox & discover 2025 미디어데이가 7일 오전 서울 서초구 삼성 강남에서 열려 신제품들이 취재진에 공개되고 있으며 발로 뛰는 더팩트는 24시간 여러분의 제보를 기다립니다.";
  // const score = 9;
  // const isHojae = score > 0;
  // const newsImage = "https://imgnews.pstatic.net/image/629/2025/04/07/202570511743990976_20250407111213466.jpg?type=w800";
  // const pressLogo = "https://mimgnews.pstatic.net/image/upload/office_logo/629/2025/03/07/logo_629_101_20250307145723.png";
  const [isScraped, setIsScraped] = useState(false);

  useEffect(() => {
    if (data?.data?.isScraped !== undefined) {
      setIsScraped(data.data.isScraped);
    }
  },[data?.data.isScraped]);

  if (isLoading || !data) {
    return <Text>로딩 중...</Text>;
  }

  const newsInfo = { ...data?.data.newsInfo };

  const isHojae = newsInfo?.score > 0;

  const onPressPinIcon = () => {
    setIsScraped(!isScraped);
    isScraped ? deleteNewsScrap(id) : addNewsScrap(id);
  };

  return (
    <View className="mx-8 my-2 mt-8 h-[700px] rounded-2xl bg-white p-6 shadow-md">
      <ScrollView>
        {/* <Text>뉴스 디테일 페이지 - {newsId}</Text> */}
        <View>
          <View>
            <Image
              source={{ uri: newsInfo?.pressLogo }}
              className="h-6 w-14"
              resizeMode="contain"
            />
          </View>

          <View className="flex-row items-center justify-between">
            <Text className="text-xl font-bold">{newsInfo?.title}</Text>
            {/* <Text className='max-w-[90%] text-xl font-bold'>{newsInfo?.title}</Text> */}
            {/* <AntDesign name='smileo' size={24} color={isHojae ? 'red' : 'blue'}/> */}
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
            <View>
              <View className="flex-row items-center">
                {/* <View className="flex-row flex-1 mb-4">
                  <Text>AI가 이 기사를 </Text>
                  <Text className={isHojae ? 'text-red-500 font-bold' : 'text-blue-500 font-bold'}>
                    {isHojae ? '호재' : '악재'}
                  </Text>
                  <Text>로 분류했어요.</Text>
                </View> */}
                {/* <Fontisto name='lightbulb' size={16} className='mr-2'/> */}
                {/* <Text className='text-lg font-bold'>AI 요약</Text> */}
                {/* <View className="flex-row items-center mx-4 px-3 py-1 bg-gray-200 rounded-xl">
                  <Text>{isHojae ? '호재' : '악재'}</Text>
                </View> */}
              </View>
              <Text className="mt-2 text-base">{newsInfo.newsSummary}</Text>
            </View>
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
  );
}
