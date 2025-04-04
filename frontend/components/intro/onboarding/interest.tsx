import { Ionicons } from '@expo/vector-icons';
import { useAllStockListQuery, useStockInterestMutation } from 'api/stock/query';
import { useUserRoleMutation } from 'api/user/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomButton from 'components/CustomButton';
import { useState } from 'react';
import {
  FlatList,
  TextInput,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
} from 'react-native';

export default function Interest() {
  const [searchText, setSearchText] = useState('');
  const [myList, setMyList] = useState<number[]>([0]);
  // Todo: 에러 처리
  const { data, isSuccess, isLoading } = useAllStockListQuery();
  const { mutate: changeUserRoleMutate } = useUserRoleMutation();
  const { mutate: setUserInterestStockMutate } = useStockInterestMutation();

  const toggleStock = (stockId: number) => {
    setMyList((prev) =>
      prev.includes(stockId) ? prev.filter((id) => id !== stockId) : [...prev, stockId]
    );
  };

  const onPressInterestStock = () => {
    changeUserRoleMutate();

    if (myList.length) {
      setUserInterestStockMutate({ stockIdList: myList });
    }
  };

  const filteredStocks = data?.data.stockList.filter((stock) =>
    stock.stockName.toLowerCase().includes(searchText.toLowerCase())
  );
  const selectedStocks = data?.data.stockList.filter((stock) => myList.includes(stock.stockId));

  return (
    <BlurOverlay className="w-full items-center gap-8 p-8 py-24">
      <View className="w-full rounded-lg bg-white p-4 py-8 shadow-lg shadow-black">
        <TextInput
          value={searchText}
          onChangeText={setSearchText}
          placeholder="코스피 상위 40개 종목을 검색해보세요"
          className="mb-2 w-full rounded-full border border-stroke bg-white px-4 py-3"
        />
        {selectedStocks && selectedStocks.length ? (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} className="mb-2">
            {selectedStocks.map((stock) => (
              <TouchableOpacity
                key={stock.stockId}
                onPress={() => toggleStock(stock.stockId)}
                className="mr-1 flex-row items-center justify-center gap-1 rounded-full bg-blue-200 px-4 py-2">
                <Text className="text-sm font-bold text-blue-900">{stock.stockName}</Text>
                <Ionicons name="close" size={16} color="#1e3a8a" />
              </TouchableOpacity>
            ))}
          </ScrollView>
        ) : (
          <Text className="mb-2 px-4 py-2 text-sm font-bold text-red-500">
            관심 종목을 선택해주세요!
          </Text>
        )}
        {isLoading ? (
          <View className="h-64 w-full items-center justify-center py-4">
            <ActivityIndicator size="large" color="#724EDB" />
          </View>
        ) : (
          isSuccess && (
            <View className="h-64 w-full">
              <FlatList
                data={filteredStocks}
                renderItem={({ item }) => {
                  const isSelected = myList.includes(item.stockId);
                  return (
                    <TouchableOpacity
                      onPress={() => toggleStock(item.stockId)}
                      className={`flex-row items-center justify-between rounded-lg border-b-[0.5px] border-b-stroke px-4 py-3 ${isSelected ? 'bg-blue-100' : ''}`}>
                      <View>
                        <Text className="text-base font-bold">{item.stockName}</Text>
                        <Text className="text-xs text-text_gray">KOSPI</Text>
                      </View>
                      <Text>{item.stockCode}</Text>
                    </TouchableOpacity>
                  );
                }}
                keyExtractor={(item) => `${item.stockId}`}
                contentContainerStyle={{ paddingBottom: 20 }}
                showsVerticalScrollIndicator={false}
                className="rounded-lg border-[0.5px] border-stroke p-1"
              />
            </View>
          )
        )}
      </View>

      <CustomButton
        variant="semiRounded"
        className="h-[45px] w-[70%]"
        onPress={onPressInterestStock}>
        선택 완료
      </CustomButton>
    </BlurOverlay>
  );
}
