import { View, Text } from 'react-native';

import Dot from './ChartLegendDot';

type ChartLegendProps = {
  color: string;
  word: string;
  count: number;
};

export default function ChartLegend({ color, word, count }: ChartLegendProps) {
  return (
    <View className="mr-5 flex-row items-center gap-3">
      <Dot color={color} />
      <Text>
        {word}: {count}
      </Text>
    </View>
  );
}
