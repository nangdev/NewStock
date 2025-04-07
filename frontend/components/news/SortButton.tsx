import { View, Text, Pressable } from "react-native";
import { useState } from "react";
import { AntDesign } from "@expo/vector-icons";

type Props = {
  sort: 'score' | 'time';
  setSort: (value: 'score' | 'time') => void;
};

export default function SortButton({ sort, setSort }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const options: { label: string; value: 'score' | 'time' }[] = [
    { label: '최신순', value: 'time' },
    { label: '관련도순', value: 'score' },
  ];

  const selectedLabel = options.find(o => o.value === sort)?.label ?? '';

  return (
    <View className="relative">
      <Pressable
        onPress={() => setIsOpen(!isOpen)}
        className="flex-row items-center px-3 py-1.5 bg-purple-100 rounded-full"
      >
      
      <AntDesign name={'down'} size={14} color="#6B46C1" className="mr-2"/>
      <Text className="text-purple-700 font-medium">{selectedLabel}</Text>
      </Pressable>

      {isOpen && (
        <View className="absolute top-12 -left-4 w-28 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
          {options.map((option) => (
            <Pressable
              key={option.value}
              onPress={() => {
                setSort(option.value);
                setIsOpen(false);
              }}
              className="px-4 py-2"
            >
              <Text className="text-gray-800">{option.label}</Text>
            </Pressable>
          ))}
        </View>
      )}
    </View>
  );
}
