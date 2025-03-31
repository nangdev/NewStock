import Init from 'components/intro/onboarding/init';
import Interest from 'components/intro/onboarding/interest';
import { useState } from 'react';
import { View } from 'react-native';

export default function Onboarding() {
  const [isFirstStep, setIsFirstStep] = useState(true);

  const onPressNextStep = () => {
    setIsFirstStep(false);
  };

  return (
    <View className="flex-1 items-center justify-center p-4">
      {isFirstStep ? <Init onPressNextStep={onPressNextStep} /> : <Interest />}
    </View>
  );
}
