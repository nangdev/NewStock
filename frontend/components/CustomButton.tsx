import React from 'react';
import { TouchableOpacity, Text } from 'react-native';
import { cn } from 'utils/styles';

type ButtonProps = {
  variant?: 'rounded' | 'semiRounded' | 'ghost';
  className?: string;
  children: React.ReactNode;
} & React.ComponentProps<typeof TouchableOpacity>;

export default function CustomButton({
  variant = 'rounded',
  className,
  children,
  ...props
}: ButtonProps) {
  const buttonStyles = cn(
    'flex-row justify-center items-center',
    variant === 'rounded' && 'bg-primary rounded-full p-2',
    variant === 'semiRounded' && 'bg-primary rounded-lg px-4 py-2',
    variant === 'ghost' && 'bg-transparent p-2',
    className
  );

  const textColor = variant === 'ghost' ? 'text-primary' : 'text-white';

  return (
    <TouchableOpacity className={buttonStyles} {...props}>
      {React.Children.map(children, (child) => (
        <Text className={cn(textColor, 'font-bold')}>{child}</Text>
      ))}
    </TouchableOpacity>
  );
}
