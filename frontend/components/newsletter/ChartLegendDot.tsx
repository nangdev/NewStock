import { View } from 'react-native';

type DotProps = {
  color: string;
};

export default function Dot({ color }: DotProps) {
  return (
    <View
      className="h-3 w-3 rounded-full"
      style={{
        backgroundColor: color,
      }}
    />
  );
}
