import { BlurView } from 'expo-blur';
import { cn } from 'utils/styles';

export default function BlurOverlay({
  className,
  children,
}: {
  className: string;
  children: React.ReactNode;
}) {
  return (
    <BlurView
      intensity={10}
      tint="regular"
      className={cn('w-full overflow-hidden rounded-2xl border border-stroke p-10', className)}>
      {children}
    </BlurView>
  );
}
