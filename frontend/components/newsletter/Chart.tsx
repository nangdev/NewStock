import { View } from 'react-native';
import { PieChart } from 'react-native-gifted-charts';

import ChartLegend from './ChartLegend';

type PieGiftedChartProps = {
  keywords: {
    word: string;
    count: number;
  }[];
};

const COLORS = [
  { color: '#009FFF', gradientCenterColor: '#006DFF' },
  { color: '#93FCF8', gradientCenterColor: '#3BE9DE' },
  { color: '#BDB2FA', gradientCenterColor: '#8F80F3' },
  { color: '#FFA5BA', gradientCenterColor: '#FF7F97' },
];

export default function PieGiftedChart({ keywords }: PieGiftedChartProps) {
  const pieData = keywords.map((keyword, index) => ({
    value: keyword.count,
    ...COLORS[index],
    focused: index === 0,
  }));

  return (
    <View className="flex-row items-center justify-center gap-6">
      <PieChart data={pieData} donut showGradient sectionAutoFocus radius={60} innerRadius={30} />
      <View className="flex justify-center gap-1">
        {keywords.map((keyword, index) => (
          <ChartLegend
            key={keyword.word}
            color={COLORS[index % COLORS.length].gradientCenterColor}
            word={keyword.word}
            count={keyword.count}
          />
        ))}
      </View>
    </View>
  );
}
