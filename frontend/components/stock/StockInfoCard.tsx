import { View, Text, Image } from 'react-native';

type Props = {
  stockId: number;
  stockName: string;
  stockCode: string;
  price: number;
  changeRate: number;
  imgUrl: string;
  totalPrice: string; // 시가총액
  issuedNum: string; // 발행주식수: lstgStqt
  capital: string; // 자본금
  parValue: number; // 액면가
  listingDate: string; // 상장일자
  industry: string; // 표준산업분류코드명
  priceChanged: number;
};

export default function StockInfoCard({
  stockName,
  stockCode,
  price,
  changeRate,
  imgUrl,
  totalPrice,
  issuedNum,
  capital,
  listingDate,
  industry,
}: Props) {
  return (
    <View className="mx-8 my-2 rounded-2xl bg-white shadow-md">
      <View className="flex-row items-center p-4">
        <Image
          source={{ uri: `data:image/png;base64,${imgUrl}` }}
          className="mr-6 h-16 w-16 rounded-xl bg-gray-200"
        />
        <View className="flex-1">
          <Text className="text-base font-bold">{stockName ? stockName : 'SSAFY'}</Text>
          <Text className="text-xs text-gray-500">{stockCode}</Text>
        </View>
        <View className="mr-6 flex-1 items-end">
          <Text className="text-base font-bold">{price.toLocaleString()} 원</Text>
          <Text
            className={`mt-1 text-sm ${
              changeRate > 0 ? 'text-red-500' : changeRate === 0 ? 'text-black' : 'text-blue-500'
            }`}>
            {changeRate.toFixed(2)}%
          </Text>
        </View>
      </View>

      <View className="flex-row  p-4">
        <View className="flex-1">
          <InfoRow label="시가총액" content={totalPrice} />
          <InfoRow label="발행주식수" content={issuedNum} />
          <InfoRow label="상장일" content={listingDate} />
        </View>
        <View className="flex-1">
          <InfoRow label="자본금" content={capital} />
          <InfoRow label="분류" content={industry} />
        </View>
      </View>
    </View>
  );
}

type Item = {
  label: string;
  content: string | number;
};

const InfoRow = ({ label, content }: Item) => {
  return (
    <View className="mx-2 flex-row justify-between border-b border-gray-200 py-2">
      <Text className="text-xs">{label}</Text>
      <Text className="text-xs">{content}</Text>
    </View>
  );
};
