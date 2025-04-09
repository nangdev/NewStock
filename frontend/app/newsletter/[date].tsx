import { Ionicons } from '@expo/vector-icons';
import { useNewsletterListQuery } from 'api/newsletter/query';
import { useAllUserStockListQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { Text, View, ScrollView, TouchableOpacity, Image, ActivityIndicator } from 'react-native';
import Markdown from 'react-native-markdown-display';
import { VictoryPie, VictoryTheme } from 'victory-native';

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
      <CustomHeader />
      <View className="px-6 pt-20">
        <Text className="mb-4 mt-2 text-center text-xl font-bold">{stockName} 뉴스레터</Text>

        <View className="mb-8 rounded-lg border p-4">
          <View className="mb-6 items-center">
            <VictoryPie
              data={currentNewsletter.keywordList.map((keyword) => ({
                x: keyword.word,
                y: keyword.count,
              }))}
              theme={VictoryTheme.clean}
              innerRadius={55}
              width={200}
              height={200}
            />
          </View>

          <View className="mb-6 flex-row flex-wrap gap-2">
            {currentNewsletter.keywordList.map((keyword) => (
              <View key={keyword.word} className="rounded-full px-2 py-1">
                <Text className="text-xs text-gray-600">
                  #{keyword.word} ({keyword.count})
                </Text>
              </View>
            ))}
          </View>

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
