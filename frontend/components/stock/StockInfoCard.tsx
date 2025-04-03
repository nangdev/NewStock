import { View, Text, Image } from "react-native";

type Props = {
  // stockId: number;
  stockName: string;
  stockCode: string;
  price: number;
  changeRate: number;
  imgUrl: string;

};

export default function StockInfoCard ({
  // stockId,
  stockName,
  stockCode,
  price,
  changeRate,
  imgUrl,
}:Props) {
  console.log(stockCode)
  return (
    <View className="bg-white rounded-2xl mx-8 my-2 shadow-md">
      <View className="flex-row items-center p-4">
        <Image
          source={{ uri: `data:image/png;base64,${imgUrl}` }}
          className="w-16 h-16 rounded-xl mr-6 bg-gray-200"
        />
        <View className="flex-1">
          <Text className="text-base font-bold">{stockName ? stockName : 'SSAFY'}</Text>
          <Text className="text-xs text-gray-500">{stockCode}</Text>
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

      <View className="flex-row  p-4">
        <View className="flex-1">
          <InfoRow label="시가총액" content={200} />
          <InfoRow label="기업명" content={200} />
          <InfoRow label="상장일" content={200} />
          <InfoRow label="홈페이지" content={200} />
        </View>
        <View className="flex-1">
          <InfoRow label="실제기업가치" content={3200} />
          <InfoRow label="대표이사" content={200} />
          <InfoRow label="발행주식수" content={200} />
        </View>
      </View>
      
    </View>
  );
}

//--------컴포넌트---------

type Item = {
  label: string;
  content: string | number;
}

const InfoRow = ({label, content}: Item) => {
  content = typeof content === 'number' ? content.toLocaleString() : content;
  return (
    <View className="flex-row border-b border-gray-200 justify-between py-2 mx-2">
      <Text className="text-xs">{label}</Text>
      <Text className="text-xs">{content}</Text>
    </View>
  );
};