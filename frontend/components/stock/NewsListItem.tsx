import { AntDesign } from '@expo/vector-icons';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import Entypo from '@expo/vector-icons/Entypo';
import PositiveIcon from 'assets/icons/positive.svg';
import NegativeIcon from 'assets/icons/negative.svg';

type Props = {
  newsId: number;
  title: string;
  description: string;
  score: number;
  publishedDate: string;
};

function NewsListItem({ newsId, title, score, publishedDate }: Props) {
  const router = useRouter();

  const onPressNewsItem = () => {
    router.navigate({
      pathname: ROUTE.NEWS.DETAIL(newsId),
      params: { newsId },
    });
  };

  return (
    <TouchableOpacity key={newsId} className="flex-row items-center p-3" onPress={onPressNewsItem}>
      <View className="mr-1 self-center">
        {score > 0 ? (
          <PositiveIcon width={23} height={23} fill="#f30606" />
        ) : (
          <NegativeIcon width={23} height={23} fill="#0658ca" />
        )}
      </View>

      <View className="flex-1 flex-row items-center justify-between">
        <Text className="flex-1 text-sm" numberOfLines={1} ellipsizeMode="tail">
          {title}
        </Text>
        <Text className="ml-4 mr-2 text-right text-xs text-gray-500">
          {toFormattedDate(publishedDate)}
        </Text>
      </View>
    </TouchableOpacity>
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
