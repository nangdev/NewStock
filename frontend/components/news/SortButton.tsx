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
        className="h-[30px] flex-row items-center justify-center rounded-full bg-blue-100 px-3 py-1 pl-4">
        <Text className="mr-1 text-sm font-medium text-blue-800">{selectedLabel}</Text>
        <AntDesign name="down" size={12} color="#1d4ed8" />
      </Pressable>

      {isOpen && (
        <View className="absolute top-[30px] z-50 mr-1 w-full rounded-lg border border-gray-100 bg-white shadow-md">
          {options.map((option) => {
            const isSelected = option.value === sort;
            return (
              <Pressable
                key={option.value}
                onPress={() => {
                  setSort(option.value);
                  setIsOpen(false);
                }}
                className={`rounded-md px-3 py-1.5 ${
                  isSelected ? 'bg-blue-50		 font-semibold text-blue-700' : 'text-gray-800'
                } hover:bg-blue-100`}>
                <Text
                  className={`text-sm ${
                    isSelected ? 'font-semibold text-blue-700' : 'text-gray-800'
                  }`}>
                  {option.label}
                </Text>
              </Pressable>
            );
          })}
        </View>
      )}
    </View>
  );
}
