import { Ionicons } from '@expo/vector-icons';
import { useNewsletterListQuery } from 'api/newsletter/query';
import { useAllUserStockListQuery } from 'api/stock/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import PieGiftedChart from 'components/newsletter/Chart';
import { useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { Text, View, ScrollView, TouchableOpacity, Image, ActivityIndicator } from 'react-native';
import Markdown from 'react-native-markdown-display';

export default function NewsLetter() {
  const { date }: { date: string } = useLocalSearchParams();
  const { data, isError, isLoading } = useNewsletterListQuery({ date });
  const { data: userStockData } = useAllUserStockListQuery();
  const [currentIndex, setCurrentIndex] = useState(0);

  if (isLoading)
    return (
      <View className="h-full w-full items-center justify-center">
        <ActivityIndicator size="large" color="#724EDB" />
      </View>
    );
  if (isError || !data?.data.newsletterList.length) {
    return (
      <>
        <CustomHeader />
        <View className="h-full w-full items-center justify-center gap-8">
          <View className="items-center">
            <Image
              source={require('../../assets/image/no_data.png')}
              style={{ width: 50, height: 50, resizeMode: 'contain' }}
            />
            <Text style={{ color: '#8A96A3' }}>해당 날짜의 뉴스 레터가 없어요 !</Text>
          </View>
        </View>
        <CustomFooter />
      </>
    );
  }

  const newsletterList = data.data.newsletterList;
  const currentNewsletter = newsletterList[currentIndex];
  const currentKeywordList = currentNewsletter.keywordList.slice(0, 6);

  const handlePrev = () => {
    if (currentIndex === 0) return;
    setCurrentIndex(currentIndex - 1);
  };

  const handleNext = () => {
    if (currentIndex === newsletterList.length - 1) return;
    setCurrentIndex(currentIndex + 1);
  };

  const stockName = userStockData?.data.stockList.find(
    (stock) => stock.stockId === currentNewsletter.stockId
  )?.stockName;

  return (
    <>
      <CustomHeader title={stockName} />
      <View className="px-6 pt-20">
        <View className="mb-4 rounded-lg p-4">
          <BlurOverlay className="w-full items-center gap-8 p-8 py-12">
            <PieGiftedChart keywords={currentKeywordList} />
          </BlurOverlay>

          <View style={{ height: 200 }}>
            <ScrollView>
              <Markdown>{currentNewsletter.content}</Markdown>
            </ScrollView>
          </View>
        </View>
        <View className="flex-row items-center justify-center gap-8">
          <TouchableOpacity onPress={handlePrev} disabled={currentIndex === 0}>
            <Ionicons name="chevron-back" size={26} color={currentIndex === 0 ? 'gray' : 'black'} />
          </TouchableOpacity>
          <Text className="text-xl font-bold">{currentIndex + 1}</Text>
          <TouchableOpacity
            onPress={handleNext}
            disabled={currentIndex === newsletterList.length - 1}>
            <Ionicons
              name="chevron-forward"
              size={26}
              color={currentIndex === newsletterList.length - 1 ? 'gray' : 'black'}
            />
          </TouchableOpacity>
        </View>
      </View>
      <CustomFooter />
    </>
  );
}
