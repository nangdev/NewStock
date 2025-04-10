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
      <View className="gap-4 rounded-lg px-6 pt-24">
        <View
          className="w-full items-center gap-8 rounded-lg border border-stroke bg-white p-8 shadow-lg"
          style={{
            backgroundColor: 'white',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
            elevation: 5,
          }}>
          <Text className="font-bold">
            <Text className="text-primary">{stockName}</Text>에서는 어떤 키워드가 등장했을까요?
          </Text>
          <PieGiftedChart keywords={currentKeywordList} />
          <View className="h-[385px] w-full items-center justify-center gap-4 border-t border-dashed border-gray-300 pt-6">
            <Text className="font-bold">
              오늘의 <Text className="text-primary">{stockName}</Text>에서는..?
            </Text>

            <ScrollView className="w-full" showsVerticalScrollIndicator={false}>
              <Markdown
                style={{
                  list_item: {
                    flexDirection: 'row',
                    alignItems: 'flex-start',
                    marginBottom: 14,
                  },
                  bullet_list_icon: {
                    width: 3,
                    height: 3,
                    borderRadius: 3,
                    backgroundColor: 'black',
                    marginTop: 10,
                    marginRight: 7,
                  },
                  bullet_list_content: {
                    flex: 1,
                    fontSize: 14,
                    lineHeight: 24,
                  },
                  strong: {
                    fontWeight: 'bold',
                    color: 'black',
                    backgroundColor: '#ffffcc',
                  },
                }}>
                {currentNewsletter.content}
              </Markdown>
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
