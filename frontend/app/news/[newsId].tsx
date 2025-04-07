import { useLocalSearchParams } from 'expo-router';
import { View, Text, ScrollView, Image } from 'react-native';
import { AntDesign } from '@expo/vector-icons';
import { useState } from 'react';

export default function NewsDetailPage() {
  const { newsId } = useLocalSearchParams();
  const title = "공개된 삼성전자 2025년형 'AI TV' [포토]";
  const content = "삼성전자 언박스&디스커버 2025unbox & discover 2025 미디어데이가 7일 오전 서울 서초구 삼성 강남에서 열려 신제품들이 취재진에 공개되고 있다. 사진영상기획부 발로 뛰는 더팩트는 24시간 여러분의 제보를 기다립니다.  카카오톡: 더팩트제보 검색  이메일:   뉴스 홈페이지: http://talk.tf.co.kr/bbs/report/write";
  const publishedDate = "2025-04-07 11:12:13";
  const newsSummary = "7일 오전 서울 서초구 삼성 강남에서 열린 삼성전자 언박스&디스커버 2025unbox & discover 2025 미디어데이가 7일 오전 서울 서초구 삼성 강남에서 열려 신제품들이 취재진에 공개되고 있으며 발로 뛰는 더팩트는 24시간 여러분의 제보를 기다립니다.";
  const score = 9;
  const isHojae = score > 0;
  const newsImage = "https://imgnews.pstatic.net/image/629/2025/04/07/202570511743990976_20250407111213466.jpg?type=w800";
  const pressLogo = "https://mimgnews.pstatic.net/image/upload/office_logo/629/2025/03/07/logo_629_101_20250307145723.png";

  const [isScrapped, setIsScrapped] = useState(false); 
  const onPressPinIcon = () => {
    setIsScrapped(!isScrapped);
  }
  return (
    <View>
      <Text>뉴스 디테일 페이지 - {newsId}</Text>
      <ScrollView className="bg-white rounded-2xl mx-8 my-2 mt-8 shadow-md h-[700px] p-6">
        <View>

          <View>
            <Image
              source={{uri: pressLogo}}
              className='w-14 h-6'
              resizeMode='contain'
            />
          </View>

          <View className='flex-row justify-between items-center'>
            <Text className='max-w-[90%] text-xl font-bold'>{title}</Text>
            {/* <AntDesign name='smileo' size={24} color={isHojae ? 'red' : 'blue'}/> */}
          </View>

          <View className='flex-row justify-between items-center'>
            <Text className='my-4 text-gray-500'>{publishedDate}</Text> 
            <AntDesign
              name={isScrapped ? 'pushpin' : 'pushpino'}
              size={24}
              style={{ transform: [{ scaleX: -1 }] }}
              onPress={onPressPinIcon}
            />
          </View>
          <View className="border-t border-gray-200 mb-8"></View>
        </View>

        <View className="my-4 ml-2 mr-12">
          <View className="flex-row mb-2">
            <View className="w-[2px] bg-black mr-6 rounded-sm" />

            <View>
              <View className="flex-row flex-wrap">
                <Text>AI가 이 기사를 </Text>
                <Text className={isHojae ? 'text-red-500 font-bold' : 'text-blue-500 font-bold'}>
                  {isHojae ? '호재' : '악재'}
                </Text>
                <Text>로 분류했어요.</Text>
              </View>

              {/* <Text>요약</Text> */}
              <Text className="text-base mt-2">{newsSummary}</Text>
            </View>
          </View>
        </View>

        { newsImage
          ? <View className='px-2'>
              <Image
                source={{uri: newsImage}}
                className="w-full h-60 rounded-lg"
                resizeMode='contain'
              />
            </View>
          : null
        }

        <View>
          <Text className='text-lg'>{content}</Text>
        </View>

      </ScrollView>
    </View>
  );
}
