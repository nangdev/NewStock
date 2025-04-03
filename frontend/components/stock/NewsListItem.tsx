import { View, Text, Image } from "react-native";

type Props = {
  newsId: string;
  title: string;
  description: string;
  score: number;
  publishedDate: string;
  hojaeIconUrl: string;
}


export default function NewsListItem ({
  newsId,
  title,
  description,
  score,
  publishedDate,
  hojaeIconUrl
}: Props) {

  return(
    <View key={newsId} className="flex-row items-center p-3">
      <Image
        source={{ uri: hojaeIconUrl || 'https://via.placeholder.com/36' }}
        className="w-9 h-9 rounded-md mr-4 bg-gray-200"
      />
      <View className="flex-1 flex-row justify-between items-center">
        <Text className="text-sm">{title}</Text>
        <Text className="text-xs text-gray-500">{publishedDate}</Text>
      </View>
    </View>
  );
}