import { View, Text } from "react-native";

type Props = {
  score: number;
  sentimentColor: string;
  financeScore: number;
  strategyScore: number;
  governScore: number;
  techScore: number;
  externalScore: number;
};

export default function ScoreBoard({
  score,
  sentimentColor,
  financeScore,
  strategyScore,
  governScore,
  techScore,
  externalScore,
}: Props) {
  const scores = [
    { label: "재무 및 자본", value: financeScore },
    { label: "전략 및 성장", value: strategyScore },
    { label: "경영 및 지배구조", value: governScore },
    { label: "기술 혁신", value: techScore },
    { label: "외부 환경", value: externalScore },
  ];

  const getBar = (value: number) => {
    const isPositive = value >= 0;
    const barWidth = Math.abs(value) * 5;
  
    return (
      <View className="relative h-3 w-full rounded-full bg-gray-200 overflow-hidden mt-1">
        <View
          className={`absolute top-0 h-full ${isPositive ? 'bg-red-400 left-1/2' : 'bg-blue-400 right-1/2'}`}
          style={{ width: `${barWidth}%` }}
        />
      </View>
    );
  };

  return (
    <View className="mx-8 min-h-[240px] px-4 pt-2">
      <View className="items-center mb-2">
        <Text
          className={`text-2xl font-bold ${
            score > 0 ? 'text-red-500' : score < 0 ? 'text-blue-500' : 'text-black'
          }`}
        >
          {score.toFixed(2)} / 10
        </Text>
      </View>
  
      <View className="gap-2">
        {scores.map((item, idx) => (
          <View key={idx}>
            <View className="flex-row justify-between">
              <Text className="text-xs text-gray-700">{item.label}</Text>
              <Text
                className={`text-xs ${
                  item.value > 0 ? 'text-red-500' : item.value < 0 ? 'text-blue-500' : 'text-black'
                }`}
              >
                {item.value !== 0 ? `${item.value.toFixed(2)}점` : '-'}
              </Text>
            </View>
            {getBar(item.value)}
          </View>
        ))}
      </View>
    </View>
  );
}
