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
  { color: '#7DD3FC', gradientCenterColor: '#089DDD' }, // 스카이블루
  { color: '#6EE7B7', gradientCenterColor: '#25A777' }, // 민트그린
  { color: '#FEF4CD', gradientCenterColor: '#FAC905' }, // 옐로우
  { color: '#FDC79B', gradientCenterColor: '#FB923C' }, // 코랄오렌지
  { color: '#FCA5A5', gradientCenterColor: '#F63C3C' }, // 라이트레드
  { color: '#C7B6FC', gradientCenterColor: '#8F6CF9' }, // 라벤더퍼플
];

export default function PieGiftedChart({ keywords }: PieGiftedChartProps) {
  const pieData = keywords.map((keyword, index) => ({
    value: keyword.count,
    ...COLORS[index % COLORS.length],
    focused: index === 0,
  }));

  return (
    <View className="flex-row items-center justify-center gap-6">
      <PieChart
        data={pieData}
        donut
        showGradient
        sectionAutoFocus
        radius={60}
        innerRadius={30}
        isAnimated
      />
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
