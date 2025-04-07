import React from "react";
import { View, Text, Image } from "react-native";
import { AntDesign } from "@expo/vector-icons";

type Props = {
  newsId: number;
  title: string;
  description: string;
  score: number;
  publishedDate: string;
  hojaeIconUrl: string;
}


function NewsListItem ({
  newsId,
  title,
  description,
  score,
  publishedDate,
  hojaeIconUrl
}: Props) {

  return(
    <View key={newsId} className="flex-row items-center p-3">
      {/* <Image
        source={{ uri: hojaeIconUrl || 'https://via.placeholder.com/36' }}
        className="w-9 h-9 rounded-md mr-4 bg-gray-200"
      /> */}
      {score > 0
      ? <AntDesign name="smileo" size={24} color="red" className="pr-3"/>
      : <AntDesign name="frowno" size={24} color="blue" className="pr-3"/>}
      <View className="flex-1 flex-row justify-between items-center">
        <Text className="text-sm flex-1" numberOfLines={1} ellipsizeMode="tail">{title}</Text>
        <Text className="text-xs text-right text-gray-500 ml-4 mr-2">{toFormattedDate(publishedDate)}</Text>
      </View>
    </View>
  );
}

const toFormattedDate = (date: string) => {
  const now = new Date();
  const published = new Date(date.replace(" ", "T")); // ISO 형식으로 변환
  
  const diffMs = now.getTime() - published.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);

  if (diffSec < 0) return '-';
  if (diffSec < 60) return `${diffSec}초 전`;
  if (diffMin < 60) return `${diffMin}분 전`;
  if (diffHr < 24) return `${diffHr}시간 전`;
  if (diffDay < 7) return `${diffDay}일 전`;
  return date.substring(0, 10); // "YYYY-MM-DD"

}

export default React.memo(NewsListItem);