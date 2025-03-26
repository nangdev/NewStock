import { TextInput, TextInputProps } from 'react-native';
import { cn } from 'utils/styles';

export default function InputField({
  value,
  onChangeText,
  placeholder,
  className,
  secureTextEntry = false,
}: { className?: string } & TextInputProps) {
  return (
    <TextInput
      value={value}
      onChangeText={onChangeText}
      placeholder={placeholder}
      secureTextEntry={secureTextEntry}
      className={cn(
        'rounded-xl border border-stroke bg-background p-3 shadow-lg shadow-black',
        className
      )}
    />
  );
}
