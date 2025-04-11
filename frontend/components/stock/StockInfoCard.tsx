import { View, Text, Image, Pressable, Modal } from 'react-native';
import { useState } from 'react';
import { AntDesign } from '@expo/vector-icons';

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
    <View className="mx-8 my-2 rounded-2xl bg-white shadow-md h-[215px]">
      <View className="ml-1 mt-1 flex-row items-center p-4">
        <Image
          source={{ uri: `data:image/png;base64,${imgUrl}` }}
          className="mr-6 h-16 w-16 rounded-xl bg-gray-200"
        />

        <View className="mr-2 flex-1">
          <View className="flex-row items-center justify-between">
            <Text className="text-base font-bold">{stockName ? stockName : 'SSAFY'}</Text>
            <Text className="text-base font-bold">{price.toLocaleString()} 원</Text>
          </View>

          <View className="mt-1 flex-row items-center justify-between">
            <Text className="text-xs text-gray-500">{stockCode}</Text>
            <Text
              className={`text-sm ${
                changeRate > 0 ? 'text-red-500' : changeRate === 0 ? 'text-black' : 'text-blue-500'
              }`}>
              {changeRate.toFixed(2)}%
            </Text>
          </View>
        </View>
      </View>

      <View className="mx-4 border-t border-gray-200" />

      <View className="mb-1 flex-row  p-4">
        <View className="flex-1">
          <InfoRow label="시가총액" content={totalPrice} />
          <InfoRow label="발행주식수" content={issuedNum} />
          <InfoRow label="상장일" content={listingDate} />
        </View>
        <View className="flex-1">
          <InfoRow label="자본금" content={capital} />
          <InfoRow label="분류" content={industry} />
          <View className="h-[24px]" />
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
  const [showTooltip, setShowTooltip] = useState(false);
  const [isTruncated, setIsTruncated] = useState(false);

  return (
    <View className="relative mx-2 min-h-[32px] border-b border-gray-200 py-2">
      <View className="flex-row items-start justify-between">
        <Text className="text-xs font-semibold">{label}</Text>
        <Pressable onPress={() => isTruncated && setShowTooltip(true)}>
          <Text
            className="text-right text-xs"
            numberOfLines={1}
            ellipsizeMode="tail"
            onTextLayout={(e) => {
              const isOverflowing = e.nativeEvent.lines.length > 1;
              setIsTruncated(isOverflowing);
            }}>
            {content}
          </Text>
        </Pressable>
      </View>

      {/* 말풍선 */}
      {showTooltip && (
        <View className="absolute left-0.5 top-full z-50 mt-2 w-[160px] items-end">
          <View className="relative mb-[-6px] items-end">
            <View
              className="left-[2.9px]"
              style={{
                position: 'absolute',
                top: 1,
                zIndex: 1,
                width: 0,
                height: 0,
                borderLeftWidth: 5,
                borderRightWidth: 5,
                borderBottomWidth: 5,
                borderLeftColor: 'transparent',
                borderRightColor: 'transparent',
                borderBottomColor: 'white',
              }}
            />
            <View
              style={{
                width: 0,
                height: 0,
                right: 3,
                borderLeftWidth: 11,
                borderRightWidth: 11,
                borderBottomWidth: 11,
                borderLeftColor: 'transparent',
                borderRightColor: 'transparent',
                borderBottomColor: '#BFDBFE',
              }}
            />
          </View>
          <View className="w-full rounded-lg border border-blue-200 bg-white px-3 py-2 shadow-md">
            <View className="flex-row items-start justify-between">
              <Text className="flex-1 pr-2 text-xs text-gray-800">{content}</Text>
              <Pressable onPress={() => setShowTooltip(false)} hitSlop={10}>
                <AntDesign name="close" size={12} color="#666" />
              </Pressable>
            </View>
          </View>
        </View>
      )}
    </View>
  );
};
