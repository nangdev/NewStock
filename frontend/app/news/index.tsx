import { useAllUserStockListQuery } from 'api/stock/query';
import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import useUserStore from 'store/user';

export default function NewsPage() {
  const { data, isSuccess, isLoading } = useAllUserStockListQuery();
  const route = useRouter();
  const userInfo = useUserStore();

  const onPressStock = (stockId: number) => {
    route.push(`${ROUTE.NEWS.DETAIL}/${stockId}`);
  };

  if (!isSuccess) return <Text>회원 관심 종목 조회 실패</Text>;
  if (isLoading) return <Text>회원 관심 종목 조회 중</Text>;

  return (
    <>
      <CustomHeader />
      <View className="gap-6 px-4 py-24">
        <Text className="text-lg font-bold text-text">
          {userInfo.userInfo?.nickname}님의 종목이에요
        </Text>
        <View className="flex-row flex-wrap justify-center gap-8 px-4">
          {data?.data.stockList.map((stock) => {
            return (
              // Todo: 이미지 경로 수정
              <TouchableOpacity
                key={stock.stockId}
                className="items-center"
                onPress={() => onPressStock(stock.stockId)}>
                <Image
                  style={{ width: 150, height: 150 }}
                  source={require('../../assets/image/sample_samsung.png')}
                />
                <Text className="font-bold text-text">{stock.stockName}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>
      <CustomFooter />
    </>
  );
}
