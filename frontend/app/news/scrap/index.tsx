import { useAllUserStockListQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { setParams } from 'expo-router/build/global-state/routing';
import { View, Text, Image, TouchableOpacity, ScrollView, ActivityIndicator } from 'react-native';
import useUserStore from 'store/user';
import { FlatList } from 'react-native';
import BlurOverlay from 'components/BlurOverlay';

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
      <View className="items-center justify-center py-24 pb-28">
        <Text className="mb-6 ml-4 items-center self-start px-6 text-lg font-semibold">
          <Text className="font-bold text-primary">{userInfo.userInfo?.nickname}</Text>님의
          종목이에요
        </Text>
        <BlurOverlay className="mx-4 max-h-[650px] w-[90%] rounded-2xl p-0">
          <View>
            <FlatList
              data={data?.data.stockList}
              numColumns={2}
              showsVerticalScrollIndicator={false}
              keyExtractor={(item) => item.stockId.toString()}
              contentContainerStyle={{
                paddingHorizontal: 32,
                paddingTop: 12,
              }}
              columnWrapperStyle={{ justifyContent: 'flex-start', marginTop: 24, marginBottom: 24 }}
              renderItem={({ item, index }) => {
                const isLastItem = index === data?.data.stockList.length - 1;
                const isOdd = data?.data.stockList.length % 2 === 1;
                const isLastOdd = isLastItem && isOdd;
                return (
                  <TouchableOpacity
                    onPress={() => onPressStock(item.stockCode, item.stockId)}
                    className="w-[140px] items-center"
                    style={{
                      marginRight: index % 2 === 0 && !isLastOdd ? 20 : 0,
                    }}>
                    <View
                      style={{
                        width: 110,
                        height: 110,
                        borderRadius: 12,
                        marginBottom: 6,
                        shadowColor: '#000',
                        shadowOffset: { width: 0, height: 2 },
                        shadowOpacity: 0.25,
                        shadowRadius: 4,
                        elevation: 5,
                        justifyContent: 'center',
                        alignItems: 'center',
                      }}>
                      <Image
                        source={{ uri: `data:image/png;base64,${item.imgUrl}` }}
                        style={{
                          width: 110,
                          height: 110,
                          resizeMode: 'contain',
                          borderRadius: 12,
                          marginBottom: 6,
                        }}
                      />
                    </View>
                    <Text className="text-center font-medium text-text">{item.stockName}</Text>
                  </TouchableOpacity>
                );
              }}
            />
          </View>
        </BlurOverlay>
      </View>
      <CustomFooter />
    </>
  );
}
