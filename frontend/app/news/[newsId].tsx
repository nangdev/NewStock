import { useLocalSearchParams } from 'expo-router';
import { View, Text } from 'react-native';

export default function NewsDetailPage() {
  const { newsId } = useLocalSearchParams();

  return (
    <View>
      <Text>뉴스 디테일 페이지 - {newsId}</Text>
    </View>
  );
}
