import NegativeIcon from 'assets/icons/negative.svg';
import PositiveIcon from 'assets/icons/positive.svg';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { getTimeAgo } from 'utils/date';

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
    <>
      <TouchableOpacity
        key={newsId}
        className="flex-row items-center justify-between px-3 py-2"
        onPress={onPressNewsItem}>
        {/* 아이콘 */}
        <View className="mr-2 self-center">
          {score > 0 ? (
            <PositiveIcon width={20} height={20} fill="#f30606" />
          ) : (
            <NegativeIcon width={20} height={20} fill="#0658ca" />
          )}
        </View>

        {/* 제목 + 시간 */}
        <View className="flex-1 flex-row items-center justify-between">
          <Text
            className="flex-1 pr-1 text-sm text-gray-900"
            numberOfLines={1}
            ellipsizeMode="tail">
            {title}
          </Text>
          <Text className="w-[45px] text-right text-xs text-gray-500">
            {toFormattedDate(publishedDate)}
          </Text>
        </View>
      </TouchableOpacity>
      <View className="mx-3 h-px bg-gray-200" />
    </>
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
