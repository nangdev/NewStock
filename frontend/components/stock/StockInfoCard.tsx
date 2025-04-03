import { View, Text, Image } from "react-native";

type Props = {
  stockId: number;
  stockName: string;
  // stockCode: string;
  price: number;
  changeRate: number;
  imgUrl: string;

};

export default function StockInfoCard ({
  stockId,
  stockName,
  // stockCode,
  price,
  changeRate,
  imgUrl,
}:Props) {
  
  return (
    <View className="bg-white rounded-2xl mx-8 my-2 shadow-md">
      <View className="flex-row items-center p-4">
        <Image
          source={{ uri: `data:image/png;base64,${imgUrl}` }}
          className="w-16 h-16 rounded-xl mr-6 bg-gray-200"
        />
        <View className="flex-1">
          <Text className="text-base font-bold">{stockName}</Text>
          {/* <Text className="text-xs text-gray-500">{stockCode}</Text> */}
        </View>
        <View className="flex-1 items-end mr-6">
          <Text className="text-base font-bold">{price.toLocaleString()} 원</Text>
          <Text className={`text-sm mt-1 ${changeRate > 0
            ? 'text-red-500' 
            : changeRate == 0
            ? 'text-black'
            : 'text-blue-500'}`}>
            {changeRate.toFixed(2)}%
          </Text>
        </View>
      </View>

      <View className="flex-row py-4">
        <View className="bg-black">

        </View>
      </View>
      
    </View>
  );
}