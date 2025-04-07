import { AntDesign } from '@expo/vector-icons';
import React from 'react';
import { View, Text } from 'react-native';

type Props = {
  newsId: number;
  title: string;
  description: string;
  score: number;
  publishedDate: string;
};

function NewsListItem({ newsId, title, score, publishedDate }: Props) {
  return (
    <View key={newsId} className="flex-row items-center p-3">
      {score > 0 ? (
        <AntDesign name="smileo" size={24} color="red" className="pr-3" />
      ) : (
        <AntDesign name="frowno" size={24} color="blue" className="pr-3" />
      )}
      <View className="flex-1 flex-row items-center justify-between">
        <Text className="flex-1 text-sm" numberOfLines={1} ellipsizeMode="tail">
          {title}
        </Text>
        <Text className="ml-4 mr-2 text-right text-xs text-gray-500">
          {toFormattedDate(publishedDate)}
        </Text>
      </View>
    </View>
  );
}

const toFormattedDate = (date: string) => {
  const now = new Date();
  const published = new Date(date.replace(' ', 'T')); // ISO 형식으로 변환

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
};

export default React.memo(NewsListItem);
