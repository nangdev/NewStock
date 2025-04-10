import { useAllUserStockListQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { setParams } from 'expo-router/build/global-state/routing';
import { View, Text, Image, TouchableOpacity, ScrollView, ActivityIndicator } from 'react-native';
import useUserStore from 'store/user';

export default function NewsScrapPage() {
  const { data, isError, isLoading } = useAllUserStockListQuery();
  const route = useRouter();
  const userInfo = useUserStore();

  const onPressStock = (stockCode: string, stockId: number) => {
    route.push({
      pathname: ROUTE.NEWS.SCRAP_NEWS(stockCode),
      params: { stockId },
    });
  };

  const onPressAddStock = () => {
    route.navigate(ROUTE.SET_INTEREST);
  };

  if (isError || !data?.data.stockList.length)
    return (
      <>
        <CustomHeader title="뉴스 스크랩" />
        <View className="h-full w-full items-center justify-center gap-8">
          <View className="items-center">
            <Image
              source={require('../../../assets/image/no_data.png')}
              style={{ width: 50, height: 50, resizeMode: 'contain' }}
            />
            <Text style={{ color: '#8A96A3' }}>관심 종목이 없어요</Text>
          </View>
          <TouchableOpacity onPress={onPressAddStock} className="rounded-lg bg-primary p-4">
            <Text className="text-center text-lg font-bold text-white">관심 종목 추가</Text>
          </TouchableOpacity>
        </View>
        <CustomFooter />
      </>
    );

  if (isLoading)
    return (
      <>
        <CustomHeader title="뉴스 스크랩" />
        <View className="h-full w-full items-center justify-center">
          <ActivityIndicator size="large" color="#724EDB" />
        </View>
        <CustomFooter />
      </>
    );

  return (
    <>
      <CustomHeader title="뉴스 스크랩" />
      <View className="py-24 pb-28">
        <Text className="mb-6 ml-4 items-center px-6 text-lg font-semibold">
          {userInfo.userInfo?.nickname}님의 종목이에요
        </Text>
        <ScrollView contentContainerStyle={{ paddingBottom: 40 }}>
          <View className="flex-row flex-wrap justify-start px-6">
            {data?.data.stockList.map((stock, index) => {
              const isLastOdd =
                data.data.stockList.length % 2 === 1 && index === data.data.stockList.length - 1;

              return (
                <TouchableOpacity
                  key={stock.stockId}
                  onPress={() => onPressStock(stock.stockCode, stock.stockId)}
                  className={`
            mb-8 ${isLastOdd ? '' : 'mr-4'}
            w-[140px]
            items-center
          `}>
                  <Image
                    source={{ uri: `data:image/png;base64,${stock.imgUrl}` }}
                    style={{
                      width: 110,
                      height: 110,
                      resizeMode: 'contain',
                      borderRadius: 12,
                      marginBottom: 10,
                    }}
                  />
                  <Text className="text-center font-medium text-text">{stock.stockName}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </ScrollView>
      </View>
      <CustomFooter />
    </>
  );
}
