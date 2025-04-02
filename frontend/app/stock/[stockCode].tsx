import { View, ScrollView, Text } from "react-native";
import { useLocalSearchParams } from "expo-router";

export default function StockDetail () {
  const {stockCode} = useLocalSearchParams();

  return (
    <View>
      <Text>stock detail page {stockCode}</Text>
    </View>
  );
}