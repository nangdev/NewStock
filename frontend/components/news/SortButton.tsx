import { AntDesign } from '@expo/vector-icons';
import { useState } from 'react';
import { View, Text, Pressable } from 'react-native';

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

  const selectedLabel = options.find((o) => o.value === sort)?.label ?? '';

  return (
    <View className="relative">
      <Pressable
        onPress={() => setIsOpen(!isOpen)}
        className="flex-row items-center rounded-full bg-blue-100 px-2 py-1">
        <AntDesign name="down" size={12} color="#1d4ed8" className="mr-1" />
        <Text className="text-sm font-medium text-blue-800">{selectedLabel}</Text>
      </Pressable>

      {isOpen && (
        <View className="absolute -left-4 top-12 z-50 w-28 rounded-lg border border-gray-200 bg-white shadow-lg">
          {options.map((option) => (
            <Pressable
              key={option.value}
              onPress={() => {
                setSort(option.value);
                setIsOpen(false);
              }}
              className="px-4 py-2">
              <Text className="text-gray-800">{option.label}</Text>
            </Pressable>
          ))}
        </View>
      )}
    </View>
  );
}
