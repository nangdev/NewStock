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
          {getTimeAgo(publishedDate)}
        </Text>
      </View>
    </TouchableOpacity>
  );
}

export default React.memo(NewsListItem);
