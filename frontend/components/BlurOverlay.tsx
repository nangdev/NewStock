import { BlurView } from 'expo-blur';
import { cn } from 'utils/styles';

export default function BlurOverlay({
  className,
  children,
  intensity = 10,
}: {
  className?: string;
  children: React.ReactNode;
  intensity?: number;
}) {
  return (
    <BlurView
      intensity={intensity}
      tint="regular"
      className={cn('w-full overflow-hidden rounded-2xl border border-gray-200 p-10', className)}>
      {children}
    </BlurView>
  );
}
