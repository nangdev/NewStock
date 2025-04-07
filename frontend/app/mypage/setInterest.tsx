import { Ionicons } from '@expo/vector-icons';
import { useAllStockListQuery, useStockInterestMutation } from 'api/stock/query';
import BlurOverlay from 'components/BlurOverlay';
import CustomButton from 'components/CustomButton';
import { useState } from 'react';
import { TextInput, View, Text, TouchableOpacity, ActivityIndicator, FlatList } from 'react-native';
import DraggableFlatList, { RenderItemParams } from 'react-native-draggable-flatlist';

export default function Interest() {
  const [searchText, setSearchText] = useState('');
  const [myList, setMyList] = useState<any[]>([]);

  const { data, isSuccess, isLoading } = useAllStockListQuery();
  const { mutate: setUserInterestStockMutate } = useStockInterestMutation();
  77;
  const toggleStock = (stock: any) => {
    setMyList((prev) => {
      const exists = prev.find((item) => item.stockId === stock.stockId);
      if (exists) {
        return prev.filter((item) => item.stockId !== stock.stockId);
      } else {
        return [
          ...prev,
          {
            stockId: stock.stockId,
            stockName: stock.stockName,
            stockCode: stock.stockCode,
          },
        ];
      }
    });
  };

  const onDeleteStock = (id: number) => {
    setMyList((prev) => prev.filter((item) => item.stockId !== id));
  };

  const onPressInterestStock = () => {
    if (myList.length) {
      const stockIdList = myList.map((stock) => stock.stockId);
      setUserInterestStockMutate({ stockIdList }); // PUT /stock/interest
    }
  };

  const filteredStocks = data?.data.stockList.filter((stock) =>
    stock.stockName.toLowerCase().includes(searchText.toLowerCase())
  );

  return (
    <BlurOverlay className="min-h-screen w-full items-center justify-center p-4">
      <View className="w-[85%] rounded-2xl bg-white p-6 py-8 shadow-lg shadow-black">
        {/* 검색창 */}
        <TextInput
          value={searchText}
          onChangeText={setSearchText}
          placeholder="코스피 상위 40개 종목을 검색해보세요"
          className="mb-4 w-full rounded-full border border-stroke bg-white px-4 py-3"
        />

        {/* 관심 종목 리스트 (드래그 앤 드롭) */}
        {myList.length ? (
          <DraggableFlatList
            data={myList}
            onDragEnd={({ data }) => setMyList(data)}
            keyExtractor={(item, index) => `${item.stockId}-${index}`}
            activationDistance={10}
            renderItem={({ item, drag }: RenderItemParams<any>) => (
              <TouchableOpacity
                onLongPress={drag}
                className="mb-2 flex-row items-center justify-between rounded-lg border border-gray-200 bg-blue-50 px-4 py-3">
                <View>
                  <Text className="text-base font-bold">{item.stockName}</Text>
                  <Text className="text-xs text-gray-500">{item.stockCode}</Text>
                </View>
                <TouchableOpacity onPress={() => onDeleteStock(item.stockId)}>
                  <Ionicons name="trash" size={20} color="#ef4444" />
                </TouchableOpacity>
              </TouchableOpacity>
            )}
            className="mb-4"
          />
        ) : (
          <Text className="mb-4 px-4 py-2 text-sm font-bold text-red-500">
            관심 종목을 선택해주세요!
          </Text>
        )}

        {/* 전체 종목 선택 리스트 */}
        {isLoading ? (
          <View className="h-64 w-full items-center justify-center py-4">
            <ActivityIndicator size="large" color="#724EDB" />
          </View>
        ) : (
          isSuccess && (
            <FlatList
              data={filteredStocks}
              renderItem={({ item }) => {
                const isSelected = myList.some((s) => s.stockId === item.stockId);
                return (
                  <TouchableOpacity
                    onPress={() => toggleStock(item)}
                    className={`flex-row items-center justify-between rounded-lg border-b-[0.5px] border-b-stroke px-4 py-3 ${
                      isSelected ? 'bg-blue-100' : ''
                    }`}>
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
          )
        )}
      </View>

      <CustomButton
        variant="semiRounded"
        className="mt-10 h-[45px] w-[70%]"
        onPress={onPressInterestStock}>
        선택 완료
      </CustomButton>
    </BlurOverlay>
  );
}
